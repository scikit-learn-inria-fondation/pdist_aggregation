# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=0
import os

import numpy as np

cimport numpy as np
cimport openmp

from joblib import cpu_count

from libc.math cimport floor, sqrt
from libc.stdlib cimport free, malloc

from cython cimport floating, integral
from cython.parallel cimport parallel, prange

# TODO: Set with a quick tuning, can be improved
DEF CHUNK_SIZE = 1024  # number of vectors

DEF MIN_CHUNK_SAMPLES = 20

DEF FLOAT_INF = 1e36

from sklearn.utils._cython_blas cimport (
  BLAS_Order,
  BLAS_Trans,
  ColMajor,
  NoTrans,
  RowMajor,
  Trans,
  _gemm,
)


cpdef int _openmp_effective_n_threads(n_threads=None):
    # Taken and adapted from sklearn.utils._openmp_helpers
    if os.getenv("OMP_NUM_THREADS"):
        # Fall back to user provided number of threads making it possible
        # to exceed the number of cpus.
        return openmp.omp_get_max_threads()
    else:
        return min(openmp.omp_get_max_threads(), cpu_count())

### Heaps utilities

cdef int _push(
    floating* dist,
    integral* idx,
    integral size,
    floating val,
    integral i_val,
) nogil except -1:
    """push (val, i_val) into the heap (dist, idx) of the given size"""
    cdef:
        integral current_idx, left_child_idx, right_child_idx, swap_idx

    # check if val should be in heap
    if val > dist[0]:
        return 0

    # insert val at position zero
    dist[0] = val
    idx[0] = i_val

    # descend the heap, swapping values until the max heap criterion is met
    current_idx = 0
    while True:
        left_child_idx = 2 * current_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= size:
            break
        elif right_child_idx >= size:
            if dist[left_child_idx] > val:
                swap_idx = left_child_idx
            else:
                break
        elif dist[left_child_idx] >= dist[right_child_idx]:
            if val < dist[left_child_idx]:
                swap_idx = left_child_idx
            else:
                break
        else:
            if val < dist[right_child_idx]:
                swap_idx = right_child_idx
            else:
                break

        dist[current_idx] = dist[swap_idx]
        idx[current_idx] = idx[swap_idx]

        current_idx = swap_idx

    dist[current_idx] = val
    idx[current_idx] = i_val

    return 0


cdef inline void dual_swap(
    floating* dist,
    integral* idx,
    integral i1,
    integral i2
) nogil:
    """swap the values at inex i1 and i2 of both dist and idx"""
    cdef:
        floating dtmp = dist[i1]
        integral itmp = idx[i1]

    dist[i1] = dist[i2]
    dist[i2] = dtmp

    idx[i1] = idx[i2]
    idx[i2] = itmp


cdef int _simultaneous_sort(
    floating* dist,
    integral* idx,
    integral size
) nogil except -1:
    """
    Perform a recursive quicksort on the dist array, simultaneously
    performing the same swaps on the idx array.
    """
    cdef:
        integral pivot_idx, i, store_idx
        floating pivot_val

    # in the small-array case, do things efficiently
    if size <= 1:
        pass
    elif size == 2:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
    elif size == 3:
        if dist[0] > dist[1]:
            dual_swap(dist, idx, 0, 1)
        if dist[1] > dist[2]:
            dual_swap(dist, idx, 1, 2)
            if dist[0] > dist[1]:
                dual_swap(dist, idx, 0, 1)
    else:
        # Determine the pivot using the median-of-three rule.
        # The smallest of the three is moved to the beginning of the array,
        # the middle (the pivot value) is moved to the end, and the largest
        # is moved to the pivot index.
        pivot_idx = size / 2
        if dist[0] > dist[size - 1]:
            dual_swap(dist, idx, 0, size - 1)
        if dist[size - 1] > dist[pivot_idx]:
            dual_swap(dist, idx, size - 1, pivot_idx)
            if dist[0] > dist[size - 1]:
                dual_swap(dist, idx, 0, size - 1)
        pivot_val = dist[size - 1]

        # partition indices about pivot.  At the end of this operation,
        # pivot_idx will contain the pivot value, everything to the left
        # will be smaller, and everything to the right will be larger.
        store_idx = 0
        for i in range(size - 1):
            if dist[i] < pivot_val:
                dual_swap(dist, idx, i, store_idx)
                store_idx += 1
        dual_swap(dist, idx, store_idx, size - 1)
        pivot_idx = store_idx

        # recursively sort each side of the pivot
        if pivot_idx > 1:
            _simultaneous_sort(dist, idx, pivot_idx)
        if pivot_idx + 2 < size:
            _simultaneous_sort(dist + pivot_idx + 1,
                               idx + pivot_idx + 1,
                               size - pivot_idx - 1)
    return 0

### K-NN helpers

cdef void _k_closest_on_chunk(
    const floating[:, ::1] X_train_c,      # IN
    const floating[:, ::1] X_test_c,       # IN
    const floating[::1] X_train_sq_norms,  # IN
    const floating *dist_middle_terms,     # IN
    floating *heaps_red_distances,         # IN/OUT
    integral *heaps_indices,               # IN/OUT
    integral k,                            # IN
    # ID of the first element of X_train_c
    integral X_train_idx_offset,
) nogil:
    cdef:
        integral i, j
    # Instead of computing the full pairwise squared distances matrix,
    # ||X_test_c - X_train_c||² = ||X_test_c||² - 2 X_test_c.X_train_c^T + ||X_train_c||²,
    # we only need to store the - 2 X_test_c.X_train_c^T + ||X_train_c||²
    # term since the argmin for a given sample X_test_c^{i} does not depend on
    # ||X_test_c^{i}||²

    # Careful: LDA, LDB and LDC are given for F-ordered arrays.
    # Here, we use their counterpart values as indicated in the documentation.
    # See the documentation of parameters here:
    # https://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html
    #
    # dist_middle_terms = -2 * X_test_c.dot(X_train_c.T)
    _gemm(RowMajor, NoTrans, Trans,
          X_test_c.shape[0], X_train_c.shape[0], X_test_c.shape[1],
          -2.0,
          &X_test_c[0, 0], X_test_c.shape[1],
          &X_train_c[0, 0], X_test_c.shape[1], 0.0,
          dist_middle_terms, X_train_c.shape[0])

    # Computing argmins here
    for i in range(X_test_c.shape[0]):
        for j in range(X_train_c.shape[0]):
            _push(heaps_red_distances + i * k,
                  heaps_indices + i * k,
                  k,
                  # reduced distance: - 2 X_test_c_i.X_train_c_j^T + ||X_train_c_j||²
                  dist_middle_terms[i * X_train_c.shape[0] + j] + X_train_sq_norms[j],
                  j + X_train_idx_offset)



cdef int _parallel_knn_on_X_test(
    const floating[:, ::1] X_train,       # IN
    const floating[:, ::1] X_test,        # IN
    const floating[::1] X_train_sq_norms, # IN
    integral chunk_size,
    integral effective_n_threads,
    integral[:, ::1] knn_indices,         # OUT
    floating[:, ::1] knn_red_distances,   # OUT
) nogil except -1:
    cdef:
        integral k = knn_indices.shape[1]
        integral d = X_test.shape[1]
        integral sf = sizeof(floating)
        integral si = sizeof(integral)
        integral n_samples_chunk = max(MIN_CHUNK_SAMPLES, chunk_size)

        integral n_train = X_train.shape[0]
        integral X_train_n_samples_chunk = min(n_train, n_samples_chunk)
        integral X_train_n_full_chunks = n_train / X_train_n_samples_chunk
        integral X_train_n_samples_rem = n_train % X_train_n_samples_chunk

        integral n_test = X_test.shape[0]
        integral X_test_n_samples_chunk = min(n_test, n_samples_chunk)
        integral X_test_n_full_chunks = n_test // X_test_n_samples_chunk
        integral X_test_n_samples_rem = n_test % X_test_n_samples_chunk

        # Counting remainder chunk in total number of chunks
        integral X_train_n_chunks = X_train_n_full_chunks + (
            n_train != (X_train_n_full_chunks * X_train_n_samples_chunk)
        )

        integral X_test_n_chunks = X_test_n_full_chunks + (
            n_test != (X_test_n_full_chunks * X_test_n_samples_chunk)
        )

        integral num_threads = min(X_train_n_chunks, effective_n_threads)

        integral X_train_start, X_train_end, X_test_start, X_test_end
        integral X_test_chunk_idx, X_train_chunk_idx, idx, jdx

        floating *dist_middle_terms_chunks
        floating *heaps_red_distances_chunks


    with nogil, parallel(num_threads=num_threads):
        # Thread local buffers

        # Temporary buffer for the -2 * X_c.dot(X_train_c.T) term
        dist_middle_terms_chunks = <floating*> malloc(X_train_n_samples_chunk * X_test_n_samples_chunk * sf)
        heaps_red_distances_chunks = <floating*> malloc(X_test_n_samples_chunk * k * sf)

        for X_test_chunk_idx in prange(X_test_n_chunks, schedule='static'):
            # We reset the heap between X chunks (memset isn't suitable here)
            for idx in range(X_test_n_samples_chunk * k):
                heaps_red_distances_chunks[idx] = FLOAT_INF

            X_test_start = X_test_chunk_idx * X_test_n_samples_chunk
            if X_test_chunk_idx == X_test_n_chunks - 1 and X_test_n_samples_rem > 0:
                X_test_end = X_test_start + X_test_n_samples_rem
            else:
                X_test_end = X_test_start + X_test_n_samples_chunk

            for X_train_chunk_idx in range(X_train_n_chunks):
                X_train_start = X_train_chunk_idx * X_train_n_samples_chunk
                if X_train_chunk_idx == X_train_n_chunks - 1 and X_train_n_samples_rem > 0:
                    X_train_end = X_train_start + X_train_n_samples_rem
                else:
                    X_train_end = X_train_start + X_train_n_samples_chunk

                _k_closest_on_chunk(
                    X_train[X_train_start:X_train_end, :],
                    X_test[X_test_start:X_test_end, :],
                    X_train_sq_norms[X_train_start:X_train_end],
                    dist_middle_terms_chunks,
                    heaps_red_distances_chunks,
                    &knn_indices[X_test_start, 0],
                    k,
                    X_train_start
                )

            # Getting the indices of the k-closest points in
            # the sorted order
            for idx in range(X_test_end - X_test_start):
                _simultaneous_sort(
                    heaps_red_distances_chunks + idx * k,
                    &knn_indices[X_test_start + idx, 0],
                    k
                )

        # end: for X_test_chunk_idx
        free(dist_middle_terms_chunks)
        free(heaps_red_distances_chunks)

    # end: with nogil, parallel
    return n_samples_chunk


cdef int _parallel_knn_on_X_train(
    const floating[:, ::1] X_train,       # IN
    const floating[:, ::1] X_test,        # IN
    const floating[::1] X_train_sq_norms, # IN
    integral chunk_size,
    integral effective_n_threads,
    integral[:, ::1] knn_indices,         # OUT
    floating[:, ::1] knn_red_distances,   # OUT
) nogil except -1:
    cdef:
        integral k = knn_indices.shape[1]
        integral d = X_test.shape[1]
        integral sf = sizeof(floating)
        integral si = sizeof(integral)
        integral n_samples_chunk = max(MIN_CHUNK_SAMPLES, chunk_size)

        integral n_train = X_train.shape[0]
        integral X_train_n_samples_chunk = min(n_train, n_samples_chunk)
        integral X_train_n_full_chunks = n_train / X_train_n_samples_chunk
        integral X_train_n_samples_rem = n_train % X_train_n_samples_chunk

        integral n_test = X_test.shape[0]
        integral X_test_n_samples_chunk = min(n_test, n_samples_chunk)
        integral X_test_n_full_chunks = n_test // X_test_n_samples_chunk
        integral X_test_n_samples_rem = n_test % X_test_n_samples_chunk

        # Counting remainder chunk in total number of chunks
        integral X_train_n_chunks = X_train_n_full_chunks + (
            n_train != (X_train_n_full_chunks * X_train_n_samples_chunk)
        )

        integral X_test_n_chunks = X_test_n_full_chunks + (
            n_test != (X_test_n_full_chunks * X_test_n_samples_chunk)
        )

        integral num_threads = min(X_train_n_chunks, effective_n_threads)

        integral X_train_start, X_train_end, X_test_start, X_test_end
        integral X_test_chunk_idx, X_train_chunk_idx, idx, jdx

        floating *dist_middle_terms_chunks
        floating *heaps_red_distances_chunks
        integral *heaps_indices_chunks

    for X_test_chunk_idx in range(X_test_n_chunks):
        X_test_start = X_test_chunk_idx * X_test_n_samples_chunk
        if X_test_chunk_idx == X_test_n_chunks - 1 and X_test_n_samples_rem > 0:
            X_test_end = X_test_start + X_test_n_samples_rem
        else:
            X_test_end = X_test_start + X_test_n_samples_chunk

        with nogil, parallel(num_threads=num_threads):
            # Thread local buffers

            # Temporary buffer for the -2 * X_test_c.dot(X_train_c.T) term
            dist_middle_terms_chunks = <floating*> malloc(
                X_train_n_samples_chunk * X_test_n_samples_chunk * sf)
            heaps_red_distances_chunks = <floating*> malloc(
                X_test_n_samples_chunk * k * sf)
            heaps_indices_chunks = <integral*> malloc(
                X_test_n_samples_chunk * k * sf)

            # Initialising heep (memset isn't suitable here)
            for idx in range(X_test_n_samples_chunk * k):
                heaps_red_distances_chunks[idx] = FLOAT_INF
                heaps_indices_chunks[idx] = -1

            for X_train_chunk_idx in prange(X_train_n_chunks, schedule='static'):
                X_train_start = X_train_chunk_idx * X_train_n_samples_chunk
                if X_train_chunk_idx == X_train_n_chunks - 1 \
                    and X_train_n_samples_rem > 0:
                    X_train_end = X_train_start + X_train_n_samples_rem
                else:
                    X_train_end = X_train_start + X_train_n_samples_chunk

                _k_closest_on_chunk(
                    X_train[X_train_start:X_train_end, :],
                    X_test[X_test_start:X_test_end, :],
                    X_train_sq_norms[X_train_start:X_train_end],
                    dist_middle_terms_chunks,
                    heaps_red_distances_chunks,
                    heaps_indices_chunks,
                    k,
                    X_train_start,
                )

            # end: for X_train_chunk_idx
            with gil:
                # Synchronising with the main heaps
                for idx in range(X_test_end - X_test_start):
                    for jdx in range(k):
                        _push(
                            &knn_red_distances[X_test_start + idx, 0],
                            &knn_indices[X_test_start + idx, 0],
                            k,
                            heaps_red_distances_chunks[idx * k + jdx],
                            heaps_indices_chunks[idx * k + jdx],
                        )

            free(dist_middle_terms_chunks)
            free(heaps_red_distances_chunks)
            free(heaps_indices_chunks)

        # end: with nogil, parallel
        # Sortting indices of the k-nn for each query vector of X_test
        for idx in prange(n_test,schedule='static',
                          nogil=True, num_threads=num_threads):
            _simultaneous_sort(
                &knn_red_distances[idx, 0],
                &knn_indices[idx, 0],
                k,
            )

        # end: with nogil, parallel
    # end: for X_test_chunk_idx
    return X_train_n_chunks

# Python interface

def parallel_knn(
    const floating[:, ::1] X_train,
    const floating[:, ::1] X_test,
    integral k,
    integral chunk_size = CHUNK_SIZE,
    bint use_chunk_on_train = True,
):
    # TODO: we could use uint32 here, working up to 4,294,967,295 indices
    int_dtype = np.int32 if integral is int else np.int64
    float_dtype = np.float32 if floating is float else np.float64
    cdef:
        integral[:, ::1] knn_indices = np.full((X_test.shape[0], k), 0,
                                               dtype=int_dtype)
        floating[:, ::1] knn_red_distances = np.full((X_test.shape[0], k),
                                                     FLOAT_INF,
                                                     dtype=float_dtype)
        floating[::1] X_train_sq_norms = np.einsum('ij,ij->i', X_train, X_train)
        integral effective_n_threads = _openmp_effective_n_threads()

    if use_chunk_on_train:
        n_parallel_chunks = _parallel_knn_on_X_train(
            X_train, X_test, X_train_sq_norms,
            chunk_size, effective_n_threads,
            knn_indices, knn_red_distances
        )
    else:
        n_parallel_chunks = _parallel_knn_on_X_test(
            X_train, X_test, X_train_sq_norms,
            chunk_size, effective_n_threads,
            knn_indices, knn_red_distances
        )

    return np.asarray(knn_indices), n_parallel_chunks

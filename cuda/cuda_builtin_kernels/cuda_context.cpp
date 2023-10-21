/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuda_context.h"
#include <cstdlib>
#include "cuda_builtin/math/cuda_fabric.h"

namespace luisa::compute::cuda {
void *alloc_host(size_t s) {
    return malloc(s);
}

void free_host(void *ptr) {
    free(ptr);
}

void memcpy_h2h(void *dest, void *src, size_t n) {
    memcpy(dest, src, n);
}

void memset_host(void *dest, int value, size_t n) {
    if ((n % 4) > 0) {
        memset(dest, value, n);
    } else {
        const size_t num_words = n / 4;
        for (size_t i = 0; i < num_words; ++i)
            ((int *)dest)[i] = value;
    }
}

// fill memory buffer with a value: this is a faster memtile variant
// for types bigger than one byte, but requires proper alignment of dst
template<typename T>
void memtile_value_host(T *dst, T value, size_t n) {
    while (n--)
        *dst++ = value;
}

void memtile_host(void *dst, const void *src, size_t srcsize, size_t n) {
    auto dst_addr = reinterpret_cast<size_t>(dst);
    auto src_addr = reinterpret_cast<size_t>(src);

    // try memtile_value first because it should be faster, but we need to ensure proper alignment
    if (srcsize == 8 && (dst_addr & 7) == 0 && (src_addr & 7) == 0)
        memtile_value_host(reinterpret_cast<int64_t *>(dst), *reinterpret_cast<const int64_t *>(src), n);
    else if (srcsize == 4 && (dst_addr & 3) == 0 && (src_addr & 3) == 0)
        memtile_value_host(reinterpret_cast<int32_t *>(dst), *reinterpret_cast<const int32_t *>(src), n);
    else if (srcsize == 2 && (dst_addr & 1) == 0 && (src_addr & 1) == 0)
        memtile_value_host(reinterpret_cast<int16_t *>(dst), *reinterpret_cast<const int16_t *>(src), n);
    else if (srcsize == 1)
        memset(dst, *reinterpret_cast<const int8_t *>(src), n);
    else {
        // generic version
        while (n--) {
            memcpy(dst, src, srcsize);
            dst = (int8_t *)dst + srcsize;
        }
    }
}

static void array_copy_nd(void *dst, const void *src,
                          const int *dst_strides, const int *src_strides,
                          const int *const *dst_indices, const int *const *src_indices,
                          const int *shape, int ndim, int elem_size) {
    if (ndim == 1) {
        for (int i = 0; i < shape[0]; i++) {
            int src_idx = src_indices[0] ? src_indices[0][i] : i;
            int dst_idx = dst_indices[0] ? dst_indices[0][i] : i;
            const char *p = (const char *)src + src_idx * src_strides[0];
            char *q = (char *)dst + dst_idx * dst_strides[0];
            // copy element
            memcpy(q, p, elem_size);
        }
    } else {
        for (int i = 0; i < shape[0]; i++) {
            int src_idx = src_indices[0] ? src_indices[0][i] : i;
            int dst_idx = dst_indices[0] ? dst_indices[0][i] : i;
            const char *p = (const char *)src + src_idx * src_strides[0];
            char *q = (char *)dst + dst_idx * dst_strides[0];
            // recurse on next inner dimension
            array_copy_nd(q, p, dst_strides + 1, src_strides + 1, dst_indices + 1, src_indices + 1, shape + 1, ndim - 1, elem_size);
        }
    }
}

static void array_copy_to_fabric(wp::fabricarray_t<void> &dst, const void *src_data,
                                 int src_stride, const int *src_indices, int elem_size) {
    const auto *src_ptr = static_cast<const int8_t *>(src_data);

    if (src_indices) {
        // copy from indexed array
        for (size_t i = 0; i < dst.nbuckets; i++) {
            const wp::fabricbucket_t &bucket = dst.buckets[i];
            auto *dst_ptr = static_cast<int8_t *>(bucket.ptr);
            size_t bucket_size = bucket.index_end - bucket.index_start;
            for (size_t j = 0; j < bucket_size; j++) {
                int idx = *src_indices;
                memcpy(dst_ptr, src_ptr + idx * elem_size, elem_size);
                dst_ptr += elem_size;
                ++src_indices;
            }
        }
    } else {
        if (src_stride == elem_size) {
            // copy from contiguous array
            for (size_t i = 0; i < dst.nbuckets; i++) {
                const wp::fabricbucket_t &bucket = dst.buckets[i];
                size_t num_bytes = (bucket.index_end - bucket.index_start) * elem_size;
                memcpy(bucket.ptr, src_ptr, num_bytes);
                src_ptr += num_bytes;
            }
        } else {
            // copy from strided array
            for (size_t i = 0; i < dst.nbuckets; i++) {
                const wp::fabricbucket_t &bucket = dst.buckets[i];
                auto *dst_ptr = static_cast<int8_t *>(bucket.ptr);
                size_t bucket_size = bucket.index_end - bucket.index_start;
                for (size_t j = 0; j < bucket_size; j++) {
                    memcpy(dst_ptr, src_ptr, elem_size);
                    src_ptr += src_stride;
                    dst_ptr += elem_size;
                }
            }
        }
    }
}

static void array_copy_from_fabric(const wp::fabricarray_t<void> &src, void *dst_data,
                                   int dst_stride, const int *dst_indices, int elem_size) {
    auto *dst_ptr = static_cast<int8_t *>(dst_data);

    if (dst_indices) {
        // copy to indexed array
        for (size_t i = 0; i < src.nbuckets; i++) {
            const wp::fabricbucket_t &bucket = src.buckets[i];
            const auto *src_ptr = static_cast<const int8_t *>(bucket.ptr);
            size_t bucket_size = bucket.index_end - bucket.index_start;
            for (size_t j = 0; j < bucket_size; j++) {
                int idx = *dst_indices;
                memcpy(dst_ptr + idx * elem_size, src_ptr, elem_size);
                src_ptr += elem_size;
                ++dst_indices;
            }
        }
    } else {
        if (dst_stride == elem_size) {
            // copy to contiguous array
            for (size_t i = 0; i < src.nbuckets; i++) {
                const wp::fabricbucket_t &bucket = src.buckets[i];
                size_t num_bytes = (bucket.index_end - bucket.index_start) * elem_size;
                memcpy(dst_ptr, bucket.ptr, num_bytes);
                dst_ptr += num_bytes;
            }
        } else {
            // copy to strided array
            for (size_t i = 0; i < src.nbuckets; i++) {
                const wp::fabricbucket_t &bucket = src.buckets[i];
                const auto *src_ptr = static_cast<const int8_t *>(bucket.ptr);
                size_t bucket_size = bucket.index_end - bucket.index_start;
                for (size_t j = 0; j < bucket_size; j++) {
                    memcpy(dst_ptr, src_ptr, elem_size);
                    dst_ptr += dst_stride;
                    src_ptr += elem_size;
                }
            }
        }
    }
}

static void array_copy_fabric_to_fabric(wp::fabricarray_t<void> &dst, const wp::fabricarray_t<void> &src, int elem_size) {
    wp::fabricbucket_t *dst_bucket = dst.buckets;
    const wp::fabricbucket_t *src_bucket = src.buckets;
    auto *dst_ptr = static_cast<int8_t *>(dst_bucket->ptr);
    const auto *src_ptr = static_cast<const int8_t *>(src_bucket->ptr);
    size_t dst_remaining = dst_bucket->index_end - dst_bucket->index_start;
    size_t src_remaining = src_bucket->index_end - src_bucket->index_start;
    size_t total_copied = 0;

    while (total_copied < dst.size) {
        if (dst_remaining <= src_remaining) {
            // copy to destination bucket
            size_t num_elems = dst_remaining;
            size_t num_bytes = num_elems * elem_size;
            memcpy(dst_ptr, src_ptr, num_bytes);

            // advance to next destination bucket
            ++dst_bucket;
            dst_ptr = static_cast<int8_t *>(dst_bucket->ptr);
            dst_remaining = dst_bucket->index_end - dst_bucket->index_start;

            // advance source offset
            src_ptr += num_bytes;
            src_remaining -= num_elems;

            total_copied += num_elems;
        } else {
            // copy to destination bucket
            size_t num_elems = src_remaining;
            size_t num_bytes = num_elems * elem_size;
            memcpy(dst_ptr, src_ptr, num_bytes);

            // advance to next source bucket
            ++src_bucket;
            src_ptr = static_cast<const int8_t *>(src_bucket->ptr);
            src_remaining = src_bucket->index_end - src_bucket->index_start;

            // advance destination offset
            dst_ptr += num_bytes;
            dst_remaining -= num_elems;

            total_copied += num_elems;
        }
    }
}

static void array_copy_to_fabric_indexed(wp::indexedfabricarray_t<void> &dst, const void *src_data,
                                         int src_stride, const int *src_indices, int elem_size) {
    const auto *src_ptr = static_cast<const int8_t *>(src_data);

    if (src_indices) {
        // copy from indexed array
        for (size_t i = 0; i < dst.size; i++) {
            size_t src_idx = src_indices[i];
            size_t dst_idx = dst.indices[i];
            void *dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);
            memcpy(dst_ptr, src_ptr + dst_idx * elem_size, elem_size);
        }
    } else {
        // copy from contiguous/strided array
        for (size_t i = 0; i < dst.size; i++) {
            size_t dst_idx = dst.indices[i];
            void *dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);
            if (dst_ptr) {
                memcpy(dst_ptr, src_ptr, elem_size);
                src_ptr += src_stride;
            }
        }
    }
}

static void array_copy_fabric_indexed_to_fabric(wp::fabricarray_t<void> &dst, const wp::indexedfabricarray_t<void> &src, int elem_size) {
    wp::fabricbucket_t *dst_bucket = dst.buckets;
    auto *dst_ptr = static_cast<int8_t *>(dst_bucket->ptr);
    int8_t *dst_end = dst_ptr + elem_size * (dst_bucket->index_end - dst_bucket->index_start);

    for (size_t i = 0; i < src.size; i++) {
        size_t src_idx = src.indices[i];
        const void *src_ptr = fabricarray_element_ptr(src.fa, src_idx, elem_size);

        if (dst_ptr >= dst_end) {
            // advance to next destination bucket
            ++dst_bucket;
            dst_ptr = static_cast<int8_t *>(dst_bucket->ptr);
            dst_end = dst_ptr + elem_size * (dst_bucket->index_end - dst_bucket->index_start);
        }

        memcpy(dst_ptr, src_ptr, elem_size);

        dst_ptr += elem_size;
    }
}

static void array_copy_fabric_indexed_to_fabric_indexed(wp::indexedfabricarray_t<void> &dst, const wp::indexedfabricarray_t<void> &src, int elem_size) {
    for (size_t i = 0; i < src.size; i++) {
        size_t src_idx = src.indices[i];
        size_t dst_idx = dst.indices[i];

        const void *src_ptr = fabricarray_element_ptr(src.fa, src_idx, elem_size);
        void *dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);

        memcpy(dst_ptr, src_ptr, elem_size);
    }
}

static void array_copy_fabric_to_fabric_indexed(wp::indexedfabricarray_t<void> &dst, const wp::fabricarray_t<void> &src, int elem_size) {
    wp::fabricbucket_t *src_bucket = src.buckets;
    const auto *src_ptr = static_cast<const int8_t *>(src_bucket->ptr);
    const int8_t *src_end = src_ptr + elem_size * (src_bucket->index_end - src_bucket->index_start);

    for (size_t i = 0; i < dst.size; i++) {
        size_t dst_idx = dst.indices[i];
        void *dst_ptr = fabricarray_element_ptr(dst.fa, dst_idx, elem_size);

        if (src_ptr >= src_end) {
            // advance to next source bucket
            ++src_bucket;
            src_ptr = static_cast<int8_t *>(src_bucket->ptr);
            src_end = src_ptr + elem_size * (src_bucket->index_end - src_bucket->index_start);
        }

        memcpy(dst_ptr, src_ptr, elem_size);

        src_ptr += elem_size;
    }
}

static void array_copy_from_fabric_indexed(const wp::indexedfabricarray_t<void> &src, void *dst_data,
                                           int dst_stride, const int *dst_indices, int elem_size) {
    auto *dst_ptr = static_cast<int8_t *>(dst_data);

    if (dst_indices) {
        // copy to indexed array
        for (size_t i = 0; i < src.size; i++) {
            size_t idx = src.indices[i];
            if (idx < src.fa.size) {
                const void *src_ptr = fabricarray_element_ptr(src.fa, idx, elem_size);
                int dst_idx = dst_indices[i];
                memcpy(dst_ptr + dst_idx * elem_size, src_ptr, elem_size);
            } else {
                fprintf(stderr, "Warp copy error: Source index %llu is out of bounds for fabric array of size %llu",
                        (unsigned long long)idx, (unsigned long long)src.fa.size);
            }
        }
    } else {
        // copy to contiguous/strided array
        for (size_t i = 0; i < src.size; i++) {
            size_t idx = src.indices[i];
            if (idx < src.fa.size) {
                const void *src_ptr = fabricarray_element_ptr(src.fa, idx, elem_size);
                memcpy(dst_ptr, src_ptr, elem_size);
                dst_ptr += dst_stride;
            } else {
                fprintf(stderr, "Warp copy error: Source index %llu is out of bounds for fabric array of size %llu",
                        (unsigned long long)idx, (unsigned long long)src.fa.size);
            }
        }
    }
}

size_t array_copy_host(void *dst, void *src, int dst_type, int src_type, int elem_size) {
    if (!src || !dst)
        return 0;

    const void *src_data = nullptr;
    const void *src_grad = nullptr;
    void *dst_data = nullptr;
    void *dst_grad = nullptr;
    int src_ndim = 0;
    int dst_ndim = 0;
    const int *src_shape = nullptr;
    const int *dst_shape = nullptr;
    const int *src_strides = nullptr;
    const int *dst_strides = nullptr;
    const int *const *src_indices = nullptr;
    const int *const *dst_indices = nullptr;

    const wp::fabricarray_t<void> *src_fabricarray = nullptr;
    wp::fabricarray_t<void> *dst_fabricarray = nullptr;

    const wp::indexedfabricarray_t<void> *src_indexedfabricarray = nullptr;
    wp::indexedfabricarray_t<void> *dst_indexedfabricarray = nullptr;

    const int *null_indices[wp::ARRAY_MAX_DIMS] = {nullptr};

    if (src_type == wp::ARRAY_TYPE_REGULAR) {
        const wp::array_t<void> &src_arr = *static_cast<const wp::array_t<void> *>(src);
        src_data = src_arr.data;
        src_grad = src_arr.grad;
        src_ndim = src_arr.ndim;
        src_shape = src_arr.shape.dims;
        src_strides = src_arr.strides;
        src_indices = null_indices;
    } else if (src_type == wp::ARRAY_TYPE_INDEXED) {
        const wp::indexedarray_t<void> &src_arr = *static_cast<const wp::indexedarray_t<void> *>(src);
        src_data = src_arr.arr.data;
        src_ndim = src_arr.arr.ndim;
        src_shape = src_arr.shape.dims;
        src_strides = src_arr.arr.strides;
        src_indices = src_arr.indices;
    } else if (src_type == wp::ARRAY_TYPE_FABRIC) {
        src_fabricarray = static_cast<const wp::fabricarray_t<void> *>(src);
        src_ndim = 1;
    } else if (src_type == wp::ARRAY_TYPE_FABRIC_INDEXED) {
        src_indexedfabricarray = static_cast<const wp::indexedfabricarray_t<void> *>(src);
        src_ndim = 1;
    } else {
        fprintf(stderr, "Warp copy error: Invalid source array type (%d)\n", src_type);
        return 0;
    }

    if (dst_type == wp::ARRAY_TYPE_REGULAR) {
        const wp::array_t<void> &dst_arr = *static_cast<const wp::array_t<void> *>(dst);
        dst_data = dst_arr.data;
        dst_grad = dst_arr.grad;
        dst_ndim = dst_arr.ndim;
        dst_shape = dst_arr.shape.dims;
        dst_strides = dst_arr.strides;
        dst_indices = null_indices;
    } else if (dst_type == wp::ARRAY_TYPE_INDEXED) {
        const wp::indexedarray_t<void> &dst_arr = *static_cast<const wp::indexedarray_t<void> *>(dst);
        dst_data = dst_arr.arr.data;
        dst_ndim = dst_arr.arr.ndim;
        dst_shape = dst_arr.shape.dims;
        dst_strides = dst_arr.arr.strides;
        dst_indices = dst_arr.indices;
    } else if (dst_type == wp::ARRAY_TYPE_FABRIC) {
        dst_fabricarray = static_cast<wp::fabricarray_t<void> *>(dst);
        dst_ndim = 1;
    } else if (dst_type == wp::ARRAY_TYPE_FABRIC_INDEXED) {
        dst_indexedfabricarray = static_cast<wp::indexedfabricarray_t<void> *>(dst);
        dst_ndim = 1;
    } else {
        fprintf(stderr, "Warp copy error: Invalid destination array type (%d)\n", dst_type);
        return 0;
    }

    if (src_ndim != dst_ndim) {
        fprintf(stderr, "Warp copy error: Incompatible array dimensionalities (%d and %d)\n", src_ndim, dst_ndim);
        return 0;
    }

    // handle fabric arrays
    if (dst_fabricarray) {
        size_t n = dst_fabricarray->size;
        if (src_fabricarray) {
            // copy from fabric to fabric
            if (src_fabricarray->size != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            array_copy_fabric_to_fabric(*dst_fabricarray, *src_fabricarray, elem_size);
            return n;
        } else if (src_indexedfabricarray) {
            // copy from fabric indexed to fabric
            if (src_indexedfabricarray->size != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            array_copy_fabric_indexed_to_fabric(*dst_fabricarray, *src_indexedfabricarray, elem_size);
            return n;
        } else {
            // copy to fabric
            if (size_t(src_shape[0]) != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            array_copy_to_fabric(*dst_fabricarray, src_data, src_strides[0], src_indices[0], elem_size);
            return n;
        }
    } else if (dst_indexedfabricarray) {
        size_t n = dst_indexedfabricarray->size;
        if (src_fabricarray) {
            // copy from fabric to fabric indexed
            if (src_fabricarray->size != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            array_copy_fabric_to_fabric_indexed(*dst_indexedfabricarray, *src_fabricarray, elem_size);
            return n;
        } else if (src_indexedfabricarray) {
            // copy from fabric indexed to fabric indexed
            if (src_indexedfabricarray->size != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            array_copy_fabric_indexed_to_fabric_indexed(*dst_indexedfabricarray, *src_indexedfabricarray, elem_size);
            return n;
        } else {
            // copy to fabric indexed
            if (size_t(src_shape[0]) != n) {
                fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
                return 0;
            }
            array_copy_to_fabric_indexed(*dst_indexedfabricarray, src_data, src_strides[0], src_indices[0], elem_size);
            return n;
        }
    } else if (src_fabricarray) {
        // copy from fabric
        size_t n = src_fabricarray->size;
        if (size_t(dst_shape[0]) != n) {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return 0;
        }
        array_copy_from_fabric(*src_fabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size);
        return n;
    } else if (src_indexedfabricarray) {
        // copy from fabric indexed
        size_t n = src_indexedfabricarray->size;
        if (size_t(dst_shape[0]) != n) {
            fprintf(stderr, "Warp copy error: Incompatible array sizes\n");
            return 0;
        }
        array_copy_from_fabric_indexed(*src_indexedfabricarray, dst_data, dst_strides[0], dst_indices[0], elem_size);
        return n;
    }

    size_t n = 1;
    for (int i = 0; i < src_ndim; i++) {
        if (src_shape[i] != dst_shape[i]) {
            fprintf(stderr, "Warp copy error: Incompatible array shapes\n");
            return 0;
        }
        n *= src_shape[i];
    }

    array_copy_nd(dst_data, src_data,
                  dst_strides, src_strides,
                  dst_indices, src_indices,
                  src_shape, src_ndim, elem_size);

    return n;
}

static void array_fill_strided(void *data, const int *shape, const int *strides, int ndim, const void *value, int value_size) {
    if (ndim == 1) {
        char *p = (char *)data;
        for (int i = 0; i < shape[0]; i++) {
            memcpy(p, value, value_size);
            p += strides[0];
        }
    } else {
        for (int i = 0; i < shape[0]; i++) {
            char *p = (char *)data + i * strides[0];
            // recurse on next inner dimension
            array_fill_strided(p, shape + 1, strides + 1, ndim - 1, value, value_size);
        }
    }
}

static void array_fill_indexed(void *data, const int *shape, const int *strides, const int *const *indices, int ndim, const void *value, int value_size) {
    if (ndim == 1) {
        for (int i = 0; i < shape[0]; i++) {
            int idx = indices[0] ? indices[0][i] : i;
            char *p = (char *)data + idx * strides[0];
            memcpy(p, value, value_size);
        }
    } else {
        for (int i = 0; i < shape[0]; i++) {
            int idx = indices[0] ? indices[0][i] : i;
            char *p = (char *)data + idx * strides[0];
            // recurse on next inner dimension
            array_fill_indexed(p, shape + 1, strides + 1, indices + 1, ndim - 1, value, value_size);
        }
    }
}

static void array_fill_fabric(wp::fabricarray_t<void> &fa, const void *value_ptr, int value_size) {
    for (size_t i = 0; i < fa.nbuckets; i++) {
        const wp::fabricbucket_t &bucket = fa.buckets[i];
        size_t bucket_size = bucket.index_end - bucket.index_start;
        memtile_host(bucket.ptr, value_ptr, value_size, bucket_size);
    }
}

static void array_fill_fabric_indexed(wp::indexedfabricarray_t<void> &ifa, const void *value_ptr, int value_size) {
    for (size_t i = 0; i < ifa.size; i++) {
        auto idx = size_t(ifa.indices[i]);
        if (idx < ifa.fa.size) {
            void *p = fabricarray_element_ptr(ifa.fa, idx, value_size);
            memcpy(p, value_ptr, value_size);
        }
    }
}

void array_fill_host(void *arr_ptr, int arr_type, const void *value_ptr, int value_size) {
    if (!arr_ptr || !value_ptr)
        return;

    if (arr_type == wp::ARRAY_TYPE_REGULAR) {
        wp::array_t<void> &arr = *static_cast<wp::array_t<void> *>(arr_ptr);
        array_fill_strided(arr.data, arr.shape.dims, arr.strides, arr.ndim, value_ptr, value_size);
    } else if (arr_type == wp::ARRAY_TYPE_INDEXED) {
        wp::indexedarray_t<void> &ia = *static_cast<wp::indexedarray_t<void> *>(arr_ptr);
        array_fill_indexed(ia.arr.data, ia.shape.dims, ia.arr.strides, ia.indices, ia.arr.ndim, value_ptr, value_size);
    } else if (arr_type == wp::ARRAY_TYPE_FABRIC) {
        wp::fabricarray_t<void> &fa = *static_cast<wp::fabricarray_t<void> *>(arr_ptr);
        array_fill_fabric(fa, value_ptr, value_size);
    } else if (arr_type == wp::ARRAY_TYPE_FABRIC_INDEXED) {
        wp::indexedfabricarray_t<void> &ifa = *static_cast<wp::indexedfabricarray_t<void> *>(arr_ptr);
        array_fill_fabric_indexed(ifa, value_ptr, value_size);
    } else {
        fprintf(stderr, "Warp fill error: Invalid array type id %d\n", arr_type);
    }
}

}// namespace luisa::compute::cuda
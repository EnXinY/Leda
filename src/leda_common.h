#ifndef LEDA_COMMON_H
#define LEDA_COMMON_H

#include <vector>
#include <iostream>
#include <bitset>
#include <omp.h>
#include "mmio_highlevel.h"
#include "leda_common.h"

using std::cout;
using std::endl;
using std::vector;
using std::min;
using std::max;

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T> >;

struct SpElement{
    INDEX_TYPE colIdx;
    INDEX_TYPE rowIdx;
    VALUE_TYPE val;
    
    SpElement(INDEX_TYPE colidx = -1, INDEX_TYPE rowidx = -1, VALUE_TYPE value = 0.0): colIdx(colidx), rowIdx(rowidx), val(value) {}
    
    SpElement& operator=(const SpElement& sp) {
        colIdx = sp.colIdx;
        rowIdx = sp.rowIdx;
        val    = sp.val;
        return *this;
    }
};

struct Matrix_COO {
    INDEX_TYPE         M;
    INDEX_TYPE         K;
    INDEX_TYPE         nnzR;

    vector<INDEX_TYPE> ColIdx;
    vector<INDEX_TYPE> RowIdx;
    vector<INDEX_TYPE> RowIdx_copy;
    vector<VALUE_TYPE> Val;

    vector<unsigned short> mask;
    vector<vector<INDEX_TYPE> > map;

    Matrix_COO() : M(0), K(0), nnzR(0), ColIdx() , RowIdx(), Val(), mask(), map() {}
};

struct SparseSlice {
    INDEX_TYPE         sliceSize;
    INDEX_TYPE         numColSlices;
    INDEX_TYPE         numRowSlices;
    INDEX_TYPE         numSlices;

    vector<INDEX_TYPE> sliceColPtr;
    vector<INDEX_TYPE> sliceRowIdx;
    vector<Matrix_COO> sliceVal;

    SparseSlice() : sliceSize(0), numColSlices(0), numRowSlices(0), sliceColPtr(), sliceRowIdx(), sliceVal() {}
};

void Read_matrix_size(char       *filename,
                      INDEX_TYPE *M, 
                      INDEX_TYPE *K, 
                      INDEX_TYPE *nnzR,
                      INDEX_TYPE *isSymmetric
                     ) {

    mmio_info(M, K, nnzR, isSymmetric, filename);
}

void Read_matrix_2_CSR(char       *filename, 
                       const INDEX_TYPE M, 
                       const INDEX_TYPE K, 
                       const INDEX_TYPE nnzR,

                       vector<INDEX_TYPE> &RowPtr, 
                       vector<INDEX_TYPE> &ColIdx, 
                       vector<VALUE_TYPE> &Val
                      ) {

    INDEX_TYPE *RowPtr_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * (M + 1));
    INDEX_TYPE *ColIdx_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * nnzR);
    VALUE_TYPE *Val_d    = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * nnzR);

    mmio_data_csr(RowPtr_d, ColIdx_d, Val_d, filename);

    for(INDEX_TYPE i = 0; i < M + 1; ++i)
        RowPtr[i] = RowPtr_d[i];
    
    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        ColIdx[i] = ColIdx_d[i];
        Val[i]    = Val_d[i];
    }

    free(Val_d);
    free(ColIdx_d);
    free(RowPtr_d);
}

void Read_matrix_2_CSC(char       *filename, 
                       const INDEX_TYPE M, 
                       const INDEX_TYPE K, 
                       const INDEX_TYPE nnzR,

                       vector<INDEX_TYPE> &ColPtr, 
                       vector<INDEX_TYPE> &RowIdx, 
                       vector<VALUE_TYPE> &Val
                      ) {

    INDEX_TYPE *ColPtr_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * (K + 1));
    INDEX_TYPE *RowIdx_d = (INDEX_TYPE *)malloc(sizeof(INDEX_TYPE) * nnzR);
    VALUE_TYPE *Val_d    = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * nnzR);

    mmio_data_csc(ColPtr_d, RowIdx_d, Val_d, filename);

    for(INDEX_TYPE i = 0; i < K + 1; ++i)
        ColPtr[i] = ColPtr_d[i];
    
    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        RowIdx[i] = RowIdx_d[i];
        Val[i]    = Val_d[i];
    }

    free(Val_d);
    free(RowIdx_d);
    free(ColPtr_d);
}

void CSC_2_CSR(const INDEX_TYPE M,
               const INDEX_TYPE K,
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &ColPtr_CSC,
               const vector<INDEX_TYPE> &RowIdx_CSC,
               const vector<VALUE_TYPE> &Val_CSC,
               
               vector<INDEX_TYPE> &RowPtr_CSR,
               vector<INDEX_TYPE> &ColIdx_CSR,
               vector<VALUE_TYPE> &Val_CSR
              ) {

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        RowPtr_CSR[RowIdx_CSC[i] + 1]++;
    }
    
    for(INDEX_TYPE i = 0; i < M; ++i) {
        RowPtr_CSR[i + 1] += RowPtr_CSR[i];
    }
    
    vector<INDEX_TYPE> row_nnzR(M, 0);
    for(INDEX_TYPE i = 0; i < K; ++i) {
        for(INDEX_TYPE j = ColPtr_CSC[i]; j < ColPtr_CSC[i + 1]; ++j) {
            INDEX_TYPE row = RowIdx_CSC[j];
            INDEX_TYPE col = i;
            VALUE_TYPE val = Val_CSC[j];
            
            INDEX_TYPE pos = RowPtr_CSR[row] + row_nnzR[row];
            Val_CSR[pos] = val;
            ColIdx_CSR[pos] = col;
            row_nnzR[row]++;
        }
    }
}

void CSR_2_CSC(const INDEX_TYPE M, 
               const INDEX_TYPE K, 
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &RowPtr_CSR, 
               const vector<INDEX_TYPE> &ColIdx_CSR, 
               const vector<VALUE_TYPE> &Val_CSR,

               vector<INDEX_TYPE> &ColPtr_CSC,
               vector<INDEX_TYPE> &RowIdx_CSC,
               vector<VALUE_TYPE> &Val_CSC
              ) {

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        ColPtr_CSC[ColIdx_CSR[i] + 1]++;
    }

    for(INDEX_TYPE i = 0; i < K; ++i) {
        ColPtr_CSC[i + 1] += ColPtr_CSC[i];
    }

    vector<INDEX_TYPE> col_nnzR(K, 0);
    for(INDEX_TYPE i = 0; i < M; ++i) {
        for(INDEX_TYPE j = RowPtr_CSR[i]; j < RowPtr_CSR[i + 1]; ++j) {
            INDEX_TYPE row = i;
            INDEX_TYPE col = ColIdx_CSR[j];
            VALUE_TYPE val = Val_CSR[j];
            
            INDEX_TYPE pos = ColPtr_CSC[col] + col_nnzR[col];
            Val_CSC[pos] = val;
            RowIdx_CSC[pos] = row;
            col_nnzR[col]++;
        }
    }
}

void CSR_2_COO(const INDEX_TYPE M, 
               const INDEX_TYPE K, 
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &RowPtr_CSR, 
               const vector<INDEX_TYPE> &ColIdx_CSR, 
               const vector<VALUE_TYPE> &Val_CSR,

               vector<INDEX_TYPE> &RowIdx_COO,
               vector<INDEX_TYPE> &ColIdx_COO,
               vector<VALUE_TYPE> &Val_COO
             ) {

    INDEX_TYPE row = 0;
    for(INDEX_TYPE i = 0; i < M; ++i) {
        for(INDEX_TYPE j = RowPtr_CSR[i]; j < RowPtr_CSR[i + 1]; ++j) {
            RowIdx_COO[j] = row;
            ColIdx_COO[j] = ColIdx_CSR[j];
            Val_COO[j]    = Val_CSR[j];
        }

        row++;
    }
}

void CSC_2_COO(const INDEX_TYPE M, 
               const INDEX_TYPE K, 
               const INDEX_TYPE nnzR,

               const vector<INDEX_TYPE> &ColPtr_CSC, 
               const vector<INDEX_TYPE> &RowIdx_CSC, 
               const vector<VALUE_TYPE> &Val_CSC,

               vector<INDEX_TYPE> &RowIdx_COO,
               vector<INDEX_TYPE> &ColIdx_COO,
               vector<VALUE_TYPE> &Val_COO
             ) {

    INDEX_TYPE col = 0;
    for(INDEX_TYPE i = 0; i < K; ++i) {
        for(INDEX_TYPE j = ColPtr_CSC[i]; j < ColPtr_CSC[i + 1]; ++j) {
            RowIdx_COO[j] = RowIdx_CSC[j];
            ColIdx_COO[j] = col;
            Val_COO[j]    = Val_CSC[j];
        }
        col++;
    }
}

void Generate_Dense_Matrix(const INDEX_TYPE M, 
                           const INDEX_TYPE K,
                           const VALUE_TYPE Val,
                           vector<VALUE_TYPE> &Matrix_Dense,
                           bool val_n,
                           bool is_row_major = true
                          ) {
    if(is_row_major) {
        for(INDEX_TYPE mm = 0; mm < M; ++mm) {
            for(INDEX_TYPE kk = 0; kk < K; ++kk) {
                if(val_n) {
                    Matrix_Dense[mm * K + kk] = mm;
                }
                else {
                    Matrix_Dense[mm * K + kk] = Val;
                }
            }
        }
    }
    else {
        for(INDEX_TYPE kk = 0; kk < K; ++kk) {
            for(INDEX_TYPE mm = 0; mm < M; ++mm) {
                if(val_n) {
                    Matrix_Dense[mm + M * kk] = kk;
                }
                else {
                    Matrix_Dense[mm + M * kk] = (1.0 + kk) + 0.1 * (1.0 + mm);
                }
            }
        }
    }
}

INDEX_TYPE CountOnes(const unsigned short num) {
    INDEX_TYPE count = 0;
    unsigned short num_tmp = num;
    while (num_tmp) {
        count += num_tmp & 1;
        num_tmp >>= 1;
    }
    return count;
}

void SpMM_CPU_CSR(const INDEX_TYPE M,
                  const INDEX_TYPE N,
                  const INDEX_TYPE K,
                  const INDEX_TYPE nnzR,
                  const vector<INDEX_TYPE> &RowPtr_CSR,
                  const vector<INDEX_TYPE> &ColIdx_CSR,
                  const vector<VALUE_TYPE> &Val_CSR,
                  const vector<VALUE_TYPE> &Matrix_B_Dense,
                  vector<VALUE_TYPE>       &Matrix_C_Dense
                 ) {
  for(INDEX_TYPE i = 0; i < M; ++i) {
    for(INDEX_TYPE j = RowPtr_CSR[i]; j < RowPtr_CSR[i+1]; ++j) {
      for(INDEX_TYPE l = 0; l < N; ++l) {
        Matrix_C_Dense[l * M + i] += Val_CSR[j] * Matrix_B_Dense[l * K + ColIdx_CSR[j]];
      }
    }
  }
}

void SpMM_CPU_CSC(const INDEX_TYPE M,
                  const INDEX_TYPE N,
                  const INDEX_TYPE K,
                  const INDEX_TYPE nnzR,
                  const vector<INDEX_TYPE> &ColPtr_CSC,
                  const vector<INDEX_TYPE> &RowIdx_CSC,
                  const vector<VALUE_TYPE> &Val_CSC,
                  const vector<VALUE_TYPE> &Matrix_B_Dense,
                  vector<VALUE_TYPE>       &Matrix_C_Dense
                 ) {
  for(INDEX_TYPE i = 0; i < K; ++i) {
    for(INDEX_TYPE j = ColPtr_CSC[i]; j < ColPtr_CSC[i+1]; ++j) {
      for(INDEX_TYPE l = 0; l < N; ++l) {
        Matrix_C_Dense[l * M + RowIdx_CSC[j]] += Val_CSC[j] * Matrix_B_Dense[l * K + i];
      }
    }
  }
}

void SpMM_CPU_Slice(const INDEX_TYPE M, 
                    const INDEX_TYPE N, 
                    const INDEX_TYPE K,
                    const vector<SparseSlice> &Matrix_SparseSlice,
                    const vector<VALUE_TYPE>  &Matrix_B_Dense,
                    vector<VALUE_TYPE>        &Matrix_C_Dense
                   ) {

    for(INDEX_TYPE p = 0; p < Matrix_SparseSlice.size(); p++) {
        for(INDEX_TYPE j = 0; j < Matrix_SparseSlice[p].numColSlices; ++j) {
                for(INDEX_TYPE i = Matrix_SparseSlice[p].sliceColPtr[j]; i < Matrix_SparseSlice[p].sliceColPtr[j + 1]; ++i) {
                    INDEX_TYPE slicennzR = Matrix_SparseSlice[p].sliceVal[i].nnzR;

                    for(INDEX_TYPE k = 0; k < slicennzR; ++k) {

                        INDEX_TYPE r = Matrix_SparseSlice[p].sliceVal[i].RowIdx_copy[k];
                        INDEX_TYPE c = Matrix_SparseSlice[p].sliceVal[i].ColIdx[k];
                        VALUE_TYPE v = Matrix_SparseSlice[p].sliceVal[i].Val[k];
                        
                        for(INDEX_TYPE l = 0; l < N; ++l) {
                                Matrix_C_Dense[l * M + r] += v * Matrix_B_Dense[l * K + c];
                            }
                    }

                }
            }
    }
}

void Matrix_Scatter(const INDEX_TYPE M, 
                    const INDEX_TYPE K, 
                    const INDEX_TYPE nnzR,
                    
                    const vector<INDEX_TYPE> &RowIdx_COO,
                    const vector<INDEX_TYPE> &ColIdx_COO,
                    const vector<VALUE_TYPE> &Val_COO,

                    const INDEX_TYPE NUM_PE,

                    vector<Matrix_COO> &Matrix_Band_COO
                    ) {
                    
    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE p = (RowIdx_COO[i]) % NUM_PE;
        INDEX_TYPE pos = Matrix_Band_COO[p].RowIdx.size();
        Matrix_Band_COO[p].RowIdx.resize(pos + 1);
        Matrix_Band_COO[p].RowIdx_copy.resize(pos + 1);
        Matrix_Band_COO[p].ColIdx.resize(pos + 1);
        Matrix_Band_COO[p].Val.resize(pos + 1);
        Matrix_Band_COO[p].RowIdx[pos] = RowIdx_COO[i] / NUM_PE;
        Matrix_Band_COO[p].RowIdx_copy[pos] = RowIdx_COO[i];
        Matrix_Band_COO[p].ColIdx[pos] = ColIdx_COO[i];
        Matrix_Band_COO[p].Val[pos] = Val_COO[i];
        Matrix_Band_COO[p].nnzR++;  
    }


#pragma omp parallel for
    for(INDEX_TYPE i = 0; i < Matrix_Band_COO.size(); ++i) {
        INDEX_TYPE max_rownum = -1;
        INDEX_TYPE max_colnum = -1;
        for(INDEX_TYPE j = 0; j < Matrix_Band_COO[i].nnzR; ++j) {
            max_rownum = max(max_rownum, Matrix_Band_COO[i].RowIdx[j]);
            max_colnum = max(max_colnum, Matrix_Band_COO[i].ColIdx[j]);
        }
        Matrix_Band_COO[i].M = max_rownum + 1;
        Matrix_Band_COO[i].K = max_colnum + 1;
    }
}

void Create_SparseSlice(const INDEX_TYPE M, 
                        const INDEX_TYPE K, 
                        const INDEX_TYPE nnzR,

                        const INDEX_TYPE sliceSize,

                        const vector<INDEX_TYPE> &RowIdx_COO,
                        const vector<INDEX_TYPE> &ColIdx_COO,
                        const vector<VALUE_TYPE> &Val_COO,

                        SparseSlice &sliceMatrix
                        ) {
    
    INDEX_TYPE numColSlices = (K + sliceSize - 1) / sliceSize;
    INDEX_TYPE numRowSlices = (M + sliceSize - 1) / sliceSize;

    INDEX_TYPE newnumCols  = numColSlices * sliceSize;
    INDEX_TYPE newnumRows  = numRowSlices * sliceSize;
    INDEX_TYPE newnnzR     = nnzR;

    if(newnumCols != K || newnumRows != M) {
        newnnzR += (newnumCols - K);
    }

    SparseSlice sliceMatrix_temp;

    sliceMatrix_temp.numColSlices = numColSlices;
    sliceMatrix_temp.numRowSlices = numRowSlices;
    sliceMatrix_temp.sliceSize    = sliceSize;

    INDEX_TYPE numSlices         = numColSlices * numRowSlices;

    sliceMatrix_temp.numSlices    = numSlices;

    vector<INDEX_TYPE> sliceCounts(numSlices, 0);
    for (INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE row       = RowIdx_COO[i];
        INDEX_TYPE col       = ColIdx_COO[i];
        INDEX_TYPE sliceRow   = row / sliceSize;
        INDEX_TYPE sliceCol   = col / sliceSize;
        INDEX_TYPE sliceIndex = sliceCol * numRowSlices + sliceRow;
        sliceCounts[sliceIndex]++;
    }

    INDEX_TYPE numSlices_nnzR = 0;
    for(INDEX_TYPE i = 0; i < numSlices; ++i) {
        if(sliceCounts[i] != 0) numSlices_nnzR++;
    }

    sliceMatrix_temp.sliceColPtr.resize(numColSlices + 1, 0);
    sliceMatrix_temp.sliceRowIdx.resize(numSlices_nnzR, 0);

    for(INDEX_TYPE j = 0; j < numColSlices; ++j) {
        for(INDEX_TYPE i = 0; i < numRowSlices; ++i) {
            INDEX_TYPE sliceIndex = j * numRowSlices + i;
            if(sliceCounts[sliceIndex] != 0) {
                sliceMatrix_temp.sliceColPtr[j + 1] += 1;
                Matrix_COO cooElem_temp;
                cooElem_temp.M    = sliceSize;
                cooElem_temp.K    = sliceSize;
                cooElem_temp.nnzR = sliceCounts[sliceIndex];

                sliceMatrix_temp.sliceVal.push_back(cooElem_temp);
            } 
        }
    }

    for(INDEX_TYPE j = 0; j < numColSlices; ++j) {
        sliceMatrix_temp.sliceColPtr[j + 1] += sliceMatrix_temp.sliceColPtr[j];
    }
    
    vector<INDEX_TYPE> sliceOffsets(numSlices, 0);
    INDEX_TYPE offset = 0;
    for(INDEX_TYPE i = 0; i < numSlices; ++i) {
        if(sliceCounts[i] != 0) {
            sliceOffsets[i] = offset;
            offset++;
        }
    }

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE row        = RowIdx_COO[i];
        INDEX_TYPE col        = ColIdx_COO[i];
        VALUE_TYPE value      = Val_COO[i];
        INDEX_TYPE sliceRow    = row / sliceSize;
        INDEX_TYPE sliceCol    = col / sliceSize;
        INDEX_TYPE sliceIndex  = sliceCol * numRowSlices + sliceRow;
        INDEX_TYPE sliceOffset = sliceOffsets[sliceIndex];

        sliceMatrix_temp.sliceRowIdx[sliceOffset] = sliceRow;
        sliceMatrix_temp.sliceVal[sliceOffset].ColIdx.push_back(col);
        sliceMatrix_temp.sliceVal[sliceOffset].RowIdx.push_back(row);
        sliceMatrix_temp.sliceVal[sliceOffset].Val.push_back(value);
        
    }

    sliceMatrix_temp.numSlices  = sliceMatrix_temp.sliceColPtr[numColSlices];
    sliceMatrix = sliceMatrix_temp;
}

void Create_Matrix_Band_SparseSlice(const INDEX_TYPE sliceSize,
                                    const Matrix_COO &Matrix_Band_COO,
                                    SparseSlice      &Matrix_Band_Slice
                                   ) {
    INDEX_TYPE M = Matrix_Band_COO.M; 
    INDEX_TYPE K = Matrix_Band_COO.K;
    INDEX_TYPE nnzR = Matrix_Band_COO.nnzR;

    vector<INDEX_TYPE> RowIdx_COO = Matrix_Band_COO.RowIdx;
    vector<INDEX_TYPE> RowIdx_COO_copy = Matrix_Band_COO.RowIdx_copy;
    vector<INDEX_TYPE> ColIdx_COO = Matrix_Band_COO.ColIdx;
    vector<VALUE_TYPE> Val_COO = Matrix_Band_COO.Val;

    INDEX_TYPE numColSlices = (K + sliceSize - 1) / sliceSize;
    INDEX_TYPE numRowSlices = (M + sliceSize - 1) / sliceSize;

    INDEX_TYPE newnumCols  = numColSlices * sliceSize;
    INDEX_TYPE newnumRows  = numRowSlices * sliceSize;
    INDEX_TYPE newnnzR     = nnzR;

    if(newnumCols != K || newnumRows != M) {
        newnnzR += (newnumCols - K);
    }

    SparseSlice Matrix_Band_Slice_temp;

    Matrix_Band_Slice_temp.numColSlices = numColSlices;
    Matrix_Band_Slice_temp.numRowSlices = numRowSlices;
    Matrix_Band_Slice_temp.sliceSize    = sliceSize;

    INDEX_TYPE numSlices         = numColSlices * numRowSlices;

    Matrix_Band_Slice_temp.numSlices    = numSlices;
    
    vector<INDEX_TYPE> sliceCounts(numSlices, 0);

    for (INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE row       = RowIdx_COO[i];
        INDEX_TYPE col       = ColIdx_COO[i];
        INDEX_TYPE sliceRow   = row / sliceSize;
        INDEX_TYPE sliceCol   = col / sliceSize;
        INDEX_TYPE sliceIndex = sliceCol * numRowSlices + sliceRow;
        sliceCounts[sliceIndex]++;
    }

    INDEX_TYPE numSlices_nnzR = 0;
    for(INDEX_TYPE i = 0; i < numSlices; ++i) {
        if(sliceCounts[i] != 0) numSlices_nnzR++;
    }

    Matrix_Band_Slice_temp.sliceColPtr.resize(numColSlices + 1, 0);
    Matrix_Band_Slice_temp.sliceRowIdx.resize(numSlices_nnzR, 0);

    for(INDEX_TYPE j = 0; j < numColSlices; ++j) {
        for(INDEX_TYPE i = 0; i < numRowSlices; ++i) {
            INDEX_TYPE sliceIndex = j * numRowSlices + i;
            if(sliceCounts[sliceIndex] != 0) {
                Matrix_Band_Slice_temp.sliceColPtr[j + 1] += 1;

                Matrix_COO cooElem_temp;
                cooElem_temp.M    = sliceSize;
                cooElem_temp.K    = sliceSize;
                cooElem_temp.nnzR = sliceCounts[sliceIndex];
                cooElem_temp.mask.resize(sliceSize, 0);
                cooElem_temp.map.resize(sliceSize);

                Matrix_Band_Slice_temp.sliceVal.push_back(cooElem_temp);
            } 
        }
    }

    for(INDEX_TYPE j = 0; j < numColSlices; ++j) {
        Matrix_Band_Slice_temp.sliceColPtr[j + 1] += Matrix_Band_Slice_temp.sliceColPtr[j];
    }
    
    vector<INDEX_TYPE> sliceOffsets(numSlices, 0);
    INDEX_TYPE offset = 0;
    for(INDEX_TYPE i = 0; i < numSlices; ++i) {
        if(sliceCounts[i] != 0) {
            sliceOffsets[i] = offset;
            offset++;
        }
    }

    vector<INDEX_TYPE> slice_push_num(numSlices_nnzR, 0);

    for(INDEX_TYPE i = 0; i < nnzR; ++i) {
        INDEX_TYPE row         = RowIdx_COO[i];
        INDEX_TYPE row_copy    = RowIdx_COO_copy[i];
        INDEX_TYPE col         = ColIdx_COO[i];
        VALUE_TYPE value       = Val_COO[i];
        INDEX_TYPE sliceRow    = row / sliceSize;
        INDEX_TYPE sliceCol    = col / sliceSize;
        INDEX_TYPE sliceIndex  = sliceCol * numRowSlices + sliceRow;
        INDEX_TYPE sliceOffset = sliceOffsets[sliceIndex];

        Matrix_Band_Slice_temp.sliceRowIdx[sliceOffset] = sliceRow;
        Matrix_Band_Slice_temp.sliceVal[sliceOffset].ColIdx.push_back(col);
        Matrix_Band_Slice_temp.sliceVal[sliceOffset].RowIdx.push_back(row);
        
        Matrix_Band_Slice_temp.sliceVal[sliceOffset].RowIdx_copy.push_back(row_copy);
        Matrix_Band_Slice_temp.sliceVal[sliceOffset].Val.push_back(value);

        Matrix_Band_Slice_temp.sliceVal[sliceOffset].mask[col - sliceCol * Slice_SIZE] |=  (0x1 << (row - sliceRow * Slice_SIZE));
        Matrix_Band_Slice_temp.sliceVal[sliceOffset].map[col - sliceCol * Slice_SIZE].push_back(slice_push_num[sliceOffset]);
        slice_push_num[sliceOffset]++;
    }

    Matrix_Band_Slice_temp.numSlices  = Matrix_Band_Slice_temp.sliceColPtr[numColSlices];
    Matrix_Band_Slice = Matrix_Band_Slice_temp;
}


void Create_Matrix_Band_SparseSlice_ex(const vector<Matrix_COO> &Matrix_Band_COO,
                                       vector<SparseSlice> &Matrix_Band_Slice) {
#pragma omp parallel for
    for(INDEX_TYPE i = 0; i < Matrix_Band_Slice.size(); ++i) {
        Create_Matrix_Band_SparseSlice(Slice_SIZE, Matrix_Band_COO[i], Matrix_Band_Slice[i]);
    }

}

void Slice_MiniSimilar_Column_reorder(Matrix_COO &sliceVal) {

    vector<INDEX_TYPE> RowIdx_tmp;
    vector<INDEX_TYPE> RowIdx_copy_tmp;
    vector<INDEX_TYPE> ColIdx_tmp;
    vector<VALUE_TYPE> Val_tmp;
    
    vector<unsigned short> mask_tmp = sliceVal.mask;

    vector<INDEX_TYPE> list;

    INDEX_TYPE mask_num = 0;

    for(INDEX_TYPE maskcol = 0; maskcol < Slice_SIZE; ++maskcol) {
        if(CountOnes(mask_tmp[maskcol]) != 0) {
            mask_num++;
        }
    }

    for(INDEX_TYPE maskcol = 0; maskcol < Slice_SIZE; ++maskcol) {
        if(CountOnes(mask_tmp[maskcol]) != 0) {
            list.push_back(maskcol);
            break;
        }
    }

    INDEX_TYPE num = 0;
    while(num < mask_num - 1) {
        INDEX_TYPE pos = list.size();
        INDEX_TYPE this_col = list[pos - 1];
        unsigned short this_mask = mask_tmp[this_col];

        INDEX_TYPE min_val = 10000;
        INDEX_TYPE min_colidx;

        for(INDEX_TYPE maskcol_next = 0; maskcol_next < Slice_SIZE; ++maskcol_next) {
            if(this_col != maskcol_next) {
                if(CountOnes(mask_tmp[maskcol_next]) != 0) {
                    INDEX_TYPE countone = CountOnes(this_mask & mask_tmp[maskcol_next]);
                    if(countone < min_val) {
                        min_val = countone;
                        min_colidx = maskcol_next;
                    }
                }
            }
        }

        list.resize(pos + 1);
        list[pos] = min_colidx;
        mask_tmp[this_col] &= 0;
        num++;
    }

    for(INDEX_TYPE i = 0; i < list.size(); ++i) {
        for(INDEX_TYPE j = 0; j < sliceVal.map[list[i]].size(); ++j) {
            RowIdx_tmp.push_back(sliceVal.RowIdx[sliceVal.map[list[i]][j]]);
            RowIdx_copy_tmp.push_back(sliceVal.RowIdx_copy[sliceVal.map[list[i]][j]]);
            ColIdx_tmp.push_back(sliceVal.ColIdx[sliceVal.map[list[i]][j]]);
            Val_tmp.push_back(sliceVal.Val[sliceVal.map[list[i]][j]]);
        }
    }

    sliceVal.RowIdx = RowIdx_tmp;
    sliceVal.RowIdx_copy = RowIdx_copy_tmp;
    sliceVal.ColIdx = ColIdx_tmp;
    sliceVal.Val = Val_tmp;
}


void Get_tile_nnzr(const SparseSlice &Matrix_SparseSlice, vector<INDEX_TYPE> &tile_nnzr, INDEX_TYPE &tile_num) {
    for(INDEX_TYPE j = 0; j < Matrix_SparseSlice.numColSlices; ++j) {
        for(INDEX_TYPE i = Matrix_SparseSlice.sliceColPtr[j]; i < Matrix_SparseSlice.sliceColPtr[j + 1]; ++i) {
            INDEX_TYPE nnzr = Matrix_SparseSlice.sliceVal[i].nnzR;
            if(nnzr != 0) {
                INDEX_TYPE pos = tile_nnzr.size();
                tile_nnzr.resize(pos + 1);
                tile_nnzr[pos] = nnzr;
                tile_num++;
            }
        }
    }
}

void Reordering(const vector<SpElement> &temp_SpElement_list,
                vector<SpElement> &SpEelment_list,
                const INDEX_TYPE base_col_index,
                const INDEX_TYPE i_start,
                const INDEX_TYPE NUM_Row,
                const INDEX_TYPE NUM_PE,
                const INDEX_TYPE WIDTH
                ) {

    SpElement sp_empty = {-1, -1, (VALUE_TYPE)0};

    vector<SpElement> scheduled_SpElement;
    
    vector<INDEX_TYPE> sliding_window(NUM_Row, -WIDTH);
    INDEX_TYPE org_row_idx;

    for(INDEX_TYPE p = 0; p < temp_SpElement_list.size(); ++p) {
        org_row_idx = temp_SpElement_list[p].rowIdx;
        INDEX_TYPE win_row_idx = sliding_window[org_row_idx] + WIDTH;
        INDEX_TYPE insert_flag = 1;
        while(insert_flag){
            if(win_row_idx >= ((INDEX_TYPE)scheduled_SpElement.size())) {
                scheduled_SpElement.resize(win_row_idx + 1);
                scheduled_SpElement[win_row_idx] = sp_empty;
            }
            SpElement sp = scheduled_SpElement[win_row_idx];
            if(sp.rowIdx == -1 && sp.colIdx == -1 && sp.val == 0.0) {
                insert_flag = 0;
            }
            else {
                win_row_idx++;
            }
        }

        scheduled_SpElement[win_row_idx].colIdx = temp_SpElement_list[p].colIdx - base_col_index;
        scheduled_SpElement[win_row_idx].rowIdx = org_row_idx;
        scheduled_SpElement[win_row_idx].val = temp_SpElement_list[p].val;
        sliding_window[org_row_idx] = win_row_idx;
    }

    INDEX_TYPE scheduled_SpElement_size = scheduled_SpElement.size();

    if(scheduled_SpElement_size > 0) {
        SpEelment_list.resize(i_start + scheduled_SpElement_size, sp_empty);
        for(INDEX_TYPE i = 0; i < scheduled_SpElement_size; ++i) {
            SpEelment_list[i + i_start] = scheduled_SpElement[i];
        }
    }
}

void Push_SpEelment_list(const vector<SpElement> &temp_SpElement_list,
                         vector<SpElement> &SpEelment_list,
                         const INDEX_TYPE base_col_index,
                         const INDEX_TYPE i_start
                        ) {

    SpElement sp_empty = {-1, -1, (VALUE_TYPE)0};

    vector<SpElement> scheduled_SpElement;

    for(INDEX_TYPE p = 0; p < temp_SpElement_list.size(); ++p) {
        INDEX_TYPE pos = scheduled_SpElement.size();
        scheduled_SpElement.resize(pos + 1);
        scheduled_SpElement[pos].rowIdx = temp_SpElement_list[p].rowIdx;
        scheduled_SpElement[pos].colIdx = temp_SpElement_list[p].colIdx - base_col_index;
        scheduled_SpElement[pos].val = temp_SpElement_list[p].val;
    }

    INDEX_TYPE scheduled_SpElement_size = scheduled_SpElement.size();

    if(scheduled_SpElement_size > 0) {
        SpEelment_list.resize(i_start + scheduled_SpElement_size, sp_empty);
        for(INDEX_TYPE i = 0; i < scheduled_SpElement_size; ++i) {
            SpEelment_list[i + i_start] = scheduled_SpElement[i];
        }
    }
}

void Create_SpElement_list_for_all_PEs(const INDEX_TYPE NUM_PE,
                                       const INDEX_TYPE NUM_ROW,
                                       const INDEX_TYPE NUM_COLUMN,
                                       const INDEX_TYPE Slice_SIZE,
                                       const INDEX_TYPE BATCH_SIZE,

                                       vector<SparseSlice> &Matrix_Band_Slice,
                                       vector<vector<SpElement> > &SpElement_list_pes,
                                       vector<INDEX_TYPE> &SpElement_list_ptr,
                                       const INDEX_TYPE WINDOWS = 10
                                      ) {
    SpElement_list_pes.resize(NUM_PE); 

    INDEX_TYPE numColSlices_max = -1;                                  
    for(INDEX_TYPE p = 0; p < NUM_PE; ++p) {
        numColSlices_max = max(Matrix_Band_Slice[p].numColSlices, numColSlices_max);
    }

    SpElement_list_ptr.resize((numColSlices_max + BATCH_SIZE - 1) / BATCH_SIZE + 1, 0);

    vector<vector<SpElement> > temp_SpElement_list_pes(NUM_PE);
    for(INDEX_TYPE i = 0; i < (numColSlices_max + BATCH_SIZE - 1) / BATCH_SIZE; ++i) {

#pragma omp parallel for     
        for(INDEX_TYPE p = 0; p < NUM_PE; p++) {
            temp_SpElement_list_pes[p].resize(0);
            for(INDEX_TYPE slicecolidx =  BATCH_SIZE * i; slicecolidx < min(BATCH_SIZE * (i + 1), Matrix_Band_Slice[p].numColSlices); ++slicecolidx) {
                for(INDEX_TYPE j = Matrix_Band_Slice[p].sliceColPtr[slicecolidx]; j < Matrix_Band_Slice[p].sliceColPtr[slicecolidx + 1]; ++j) {
                    INDEX_TYPE slicennzR = Matrix_Band_Slice[p].sliceVal[j].nnzR;
                    Slice_MiniSimilar_Column_reorder(Matrix_Band_Slice[p].sliceVal[j]);

                    for(INDEX_TYPE k = 0; k < slicennzR; ++k) {
                        INDEX_TYPE pos = temp_SpElement_list_pes[p].size();
                        temp_SpElement_list_pes[p].resize(pos + 1);
                        temp_SpElement_list_pes[p][pos] = SpElement(Matrix_Band_Slice[p].sliceVal[j].ColIdx[k], Matrix_Band_Slice[p].sliceVal[j].RowIdx[k], Matrix_Band_Slice[p].sliceVal[j].Val[k]);
                    }
                }
            } 

            INDEX_TYPE i_start = SpElement_list_pes[p].size();
            INDEX_TYPE base_col_index = i * BATCH_SIZE * Slice_SIZE;

            Reordering(temp_SpElement_list_pes[p],
                       SpElement_list_pes[p],
                       base_col_index,
                       i_start,
                       NUM_ROW,
                       NUM_PE,
                       WINDOWS
                      );
        }

        INDEX_TYPE max_len = 0;
        for(INDEX_TYPE p = 0; p < NUM_PE; ++p) {
            max_len = max((INDEX_TYPE) SpElement_list_pes[p].size(), max_len);
        }
        
        for(INDEX_TYPE p = 0; p < NUM_PE; ++p) {
            SpElement_list_pes[p].resize(max_len, SpElement(-1, -1, 0.0));
        }
        
        SpElement_list_ptr[i + 1] = max_len;
    } 
}


void Create_SpElement_list_for_all_channels(const vector<vector<SpElement> > &SpElement_list_pes,
                                            const vector<INDEX_TYPE>         &SpElement_list_ptr,
                                            vector<vector<unsigned long, tapa::aligned_allocator<unsigned long> > > &Matrix_A_fpga_data,
                                            const INDEX_TYPE HBM_CHANNEL_A_NUM = 8
                                           ) {
    INDEX_TYPE Matrix_fpga_data_column_size = 8 * SpElement_list_ptr[SpElement_list_ptr.size() - 1] * 4 / 4;
    INDEX_TYPE Matrix_fpga_data_channel_size  = ((Matrix_fpga_data_column_size + 512 - 1) / 512) * 512;

    for(INDEX_TYPE c = 0; c < HBM_CHANNEL_A_NUM; ++c) {
        Matrix_A_fpga_data[c].resize(Matrix_fpga_data_channel_size, 0);
    }
    
    for(INDEX_TYPE i = 0; i < SpElement_list_ptr[SpElement_list_ptr.size() - 1]; ++i) {
        for(INDEX_TYPE c = 0; c < HBM_CHANNEL_A_NUM; ++c) {
            for(INDEX_TYPE j = 0; j < 8; ++j) {
                SpElement sp = SpElement_list_pes[j + c * 8][i];

                unsigned long x = 0;
                if(sp.rowIdx == -1) {
                    x = 0x3FFFF;
                    x = x << 32;
                } 
                else {
                    unsigned long x_col = sp.colIdx;
                    x_col = (x_col & 0x3FFF) << (32 + 18); 
                    unsigned long x_row = sp.rowIdx;
                    x_row = (x_row & 0x3FFFF) << 32;
                    VALUE_TYPE x_float = sp.val;
                    
                    unsigned int x_float_in_int = *((unsigned int*)(&x_float));
                    unsigned long x_float_val_64 = ((unsigned long) x_float_in_int);
                    x_float_val_64 = x_float_val_64 & 0xFFFFFFFF;

                    x = x_col | x_row | x_float_val_64;
                }
                if(HBM_CHANNEL_A_NUM * 8 <= 16) {
                    Matrix_A_fpga_data[c][j + i * 8] = x;
                } 
                else if(HBM_CHANNEL_A_NUM == 8) {


                    INDEX_TYPE seg = 16 / HBM_CHANNEL_A_NUM;
                    INDEX_TYPE seg_idx = (c * 8 + j) / seg;
                    INDEX_TYPE c_new = seg_idx % HBM_CHANNEL_A_NUM;
                    INDEX_TYPE j_new = seg * (seg_idx / HBM_CHANNEL_A_NUM) + j % seg;
                    Matrix_A_fpga_data[c_new][j_new + i * 8] = x;
                } 
                else if(HBM_CHANNEL_A_NUM == 4) {
                    INDEX_TYPE pe_idx = j + c * 8;
                    Matrix_A_fpga_data[(pe_idx / 4) % 4][pe_idx % 4 + (pe_idx / 16) * 4 + i * 8] = x;
                }
            }
        }
    }
}

void Create_SpElement_list_data_FPGA(const vector<INDEX_TYPE> &SpElement_list_ptr,
                                     aligned_vector<INDEX_TYPE> &SpElement_list_ptr_fpga
                                    ) {
    INDEX_TYPE SpElement_list_ptr_fpga_size = ((SpElement_list_ptr.size() + 15) / 16) * 16;
    INDEX_TYPE SpElement_list_ptr_fpga_chunk_size = ((SpElement_list_ptr_fpga_size + 1023) / 1024) * 1024;
    SpElement_list_ptr_fpga.resize(SpElement_list_ptr_fpga_chunk_size, 0);
    for(INDEX_TYPE i = 0; i < SpElement_list_ptr.size(); ++i) {
        SpElement_list_ptr_fpga[i] = SpElement_list_ptr[i];
    }
}

void Create_Matrix_B_data_FPGA(const INDEX_TYPE K,
                               const INDEX_TYPE N,
                               const INDEX_TYPE HBM_CHANNEL_B_NUM,
                               const vector<VALUE_TYPE> &Matrix_B_CPU_Dense,
                               vector<aligned_vector<VALUE_TYPE> > &Matrix_B_fpga_data
                              ) {
    INDEX_TYPE mat_B_fpga_column_size;

    if(HBM_CHANNEL_B_NUM == 8) {
        mat_B_fpga_column_size = ((K + 16 - 1) / 16) * 16;
    }
    else if(HBM_CHANNEL_B_NUM == 4) {
        mat_B_fpga_column_size = ((K + 8 - 1) / 8) * 8 * 2;
    }

    INDEX_TYPE mat_B_fpga_chunk_size = ((mat_B_fpga_column_size * (N / 8) + 1023)/1024) * 1024;

    for(INDEX_TYPE c = 0; c < HBM_CHANNEL_B_NUM; ++c) {
        Matrix_B_fpga_data[c].resize(mat_B_fpga_chunk_size, 0.0);
    }
    for(INDEX_TYPE nn = 0; nn < N; ++nn) {
        for(INDEX_TYPE kk = 0; kk < K; ++kk) {
            INDEX_TYPE pos = (kk / 8) * 16 + (nn % 2) * 8 + kk % 8 + mat_B_fpga_column_size * (nn / 8);     
            Matrix_B_fpga_data[(nn/2) % 4][pos] = Matrix_B_CPU_Dense[kk + K * nn];
        }
    }
}


void Create_Matrix_C_data_FPGA(const INDEX_TYPE M,
                               const INDEX_TYPE N,
                               const INDEX_TYPE HBM_CHANNEL_C_NUM,
                               const vector<VALUE_TYPE> &Matrix_C_CPU_Dense,
                               vector<aligned_vector<VALUE_TYPE> > &Matrix_C_fpga_data
                              ) {
    INDEX_TYPE mat_C_fpga_column_size = ((M + 16 - 1) / 16) * 16;
    INDEX_TYPE mat_C_fpga_chunk_size = ((mat_C_fpga_column_size * (N / 8) + 1023)/1024) * 1024;
    for(INDEX_TYPE c = 0; c < HBM_CHANNEL_C_NUM; ++c) {
        Matrix_C_fpga_data[c].resize(mat_C_fpga_chunk_size, 0.0);
    }
                              
}

void Verify_correctness(INDEX_TYPE &error_num,
                        const VALUE_TYPE &CPU_val,
                        const VALUE_TYPE &FPGA_val,
                        const double     threshold = 1e-4
                       ) {
    double difference = fabs(CPU_val - FPGA_val);
    double x = min(fabs(CPU_val), fabs(FPGA_val)) + threshold;
    if(difference / x > threshold) {
        error_num++;
    }
}

#endif

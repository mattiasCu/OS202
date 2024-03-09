#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"


namespace {
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  #pragma omp parallel for
  
  for (int j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
  for (int k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
  for (int i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
  
    
      
        C(i, j) += A(i, k) * B(k, j);
}
const int szBlock = 32;
}  // namespace


Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  const int szBlock = 512;
  const int maxLine = A.nbRows/szBlock;
  const int maxCol = A.nbCols/szBlock;
  for (int i = 0; i < maxLine; i++)
  for (int j = 0; j < maxCol; j++)
    for (int k = 0; k < std::max(maxLine,maxCol) ; k++)
      prodSubBlocks(i*szBlock,j*szBlock,k*szBlock, szBlock, A,B,C);
  //prodSubBlocks(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C);
  return C;
}

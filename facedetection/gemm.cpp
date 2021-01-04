#include "gemm.h"

void transpose(float* b, const float* a, int row, int column)
{
    int apos;
    int bpos = -1;
    int num = 0;
    for (int i = 0; i < column; i++)
    {
        apos = i;
        for (int j = 0; j < row; j++)
        {
            b[++bpos] = a[apos];
            apos += column;
        }
    }
}

void m_product_row(float* c, const float* a, const float* b, int row1, int column1, int column2) {
    int apos = 0;
    int bpos;
    int cpos = 0;
    for (int i = 0; i < row1; ++i) {
        bpos = 0;
        for (int j = 0; j < column1; ++j) {
            for (int k = 0; k < column2; ++k) {
                c[cpos++] += a[apos] * b[bpos++];
            }
            apos++;
            cpos -= column2;
        }
        cpos += column2;
    }
}
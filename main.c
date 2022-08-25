/**************************************************************************************************
* main.c: Implementerar en maskininlärningsmodell som baseras på linjär regression. Träningsdata 
*         läses in från en textfil. Modellen tränas följt av att prediktion genomförs med 100 %
*         precision, vilket indikerar lyckad träning.
*
*         Kompilera koden och skapa en körbar fil döpt main.exe med följande kommando:
*         $ gcc main.c lin_reg.c double_vector.c uint_vector.c -o main.exe -Wall
*
*         Kör sedan programmet med följande kommando:
*         $ main.exe
**************************************************************************************************/
#include "lin_reg.h"

/**************************************************************************************************
* main: Implementerar en regressionsmodell och läser in träningsdata från en fil döpt data.txt.
*       Modellen tränas under 1000 epoker med en lärhastighet på 1 %. Modellen testas sedan för 
*       insignaler inom intervallet [-10, 10] med en stegringshastighet på 1, där indata samt 
*       motsvarande predikterad utdata skrivs ut i terminalen. Resultatet indikerar prediktion 
*       med 100 % precision.
**************************************************************************************************/
int main(void)
{
   struct lin_reg l1;
   lin_reg_new(&l1);
   lin_reg_load_training_data(&l1, "data.txt");
   lin_reg_train(&l1, 1000, 0.01);
   lin_reg_predict_range(&l1, -10, 10, 1, 0.0001, stdout);
   return 0;
}
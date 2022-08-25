/**************************************************************************************************
* main.c: Implementerar en maskininl�rningsmodell som baseras p� linj�r regression. Tr�ningsdata 
*         l�ses in fr�n en textfil. Modellen tr�nas f�ljt av att prediktion genomf�rs med 100 %
*         precision, vilket indikerar lyckad tr�ning.
*
*         Kompilera koden och skapa en k�rbar fil d�pt main.exe med f�ljande kommando:
*         $ gcc main.c lin_reg.c double_vector.c uint_vector.c -o main.exe -Wall
*
*         K�r sedan programmet med f�ljande kommando:
*         $ main.exe
**************************************************************************************************/
#include "lin_reg.h"

/**************************************************************************************************
* main: Implementerar en regressionsmodell och l�ser in tr�ningsdata fr�n en fil d�pt data.txt.
*       Modellen tr�nas under 1000 epoker med en l�rhastighet p� 1 %. Modellen testas sedan f�r 
*       insignaler inom intervallet [-10, 10] med en stegringshastighet p� 1, d�r indata samt 
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
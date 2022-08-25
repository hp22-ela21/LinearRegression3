/**************************************************************************************************
* double_vector.c: Innehåller funktionsdefinitioner för implementering av dynamiska vektorer
*                  för lagring av flyttal via strukten double_vector.
**************************************************************************************************/
#include "double_vector.h"

/**************************************************************************************************
* double_vector_new: Initierar angiven vektor.
*
*                    - self: Pekare till vektorn.
**************************************************************************************************/
void double_vector_new(struct double_vector* self)
{
   self->data = 0;
   self->size = 0;
   return;
}

/**************************************************************************************************
* double_vector_delete: Tömmer innehållet i angiven vektor.
*
*                       - self: Pekare till vektorn.
**************************************************************************************************/
void double_vector_delete(struct double_vector* self)
{
   free(self->data);
   self->data = 0;
   self->size = 0;
   return;
}

/**************************************************************************************************
* double_vector_ptr_new: Returnerar en pekare till en ny heapallokerad vektor av angiven storlek.
*
*                        - size: Fältets storlek (antalet element det rymmer) vid start.
**************************************************************************************************/
struct double_vector* double_vector_ptr_new(const size_t size)
{
   struct double_vector* self = (struct double_vector*)malloc(sizeof(struct double_vector));
   if (!self) return 0;
   self->data = 0;
   self->size = 0;
   double_vector_resize(self, size);
   return self;
}

/**************************************************************************************************
* double_vector_ptr_delete: Frigör minne för angiven heapallokerad vektor. Vektorpekarens adress
*                           passeras för att både frigöra minnet för det dynamiska fält denna pekar
*                           på, minnet för själva vektorn samt att vektorpekaren sätts till null.
*
*                           - self: Adressen till vektorpekaren.
**************************************************************************************************/
void double_vector_ptr_delete(struct double_vector** self)
{
   double_vector_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* double_vector_resize: Ändrar storleken / kapaciteten på angiven vektor via omallokering.
*
*                      - self    : Pekare till vektorn.
*                      - new_size: Vektorns nya storlek efter omallokeringen.
**************************************************************************************************/
int double_vector_resize(struct double_vector* self,
                         const size_t new_size)
{
   double* copy = (double*)realloc(self->data, sizeof(double) * new_size);
   if (!copy) return 1;
   self->data = copy;
   self->size = new_size;
   return 0;
}

/**************************************************************************************************
* double_vector_push: Lägger till ett nytt element längst bak i angiven vektor.
*
*                     - self       : Pekare till vektorn.
*                     - new_element: Det nya element som skall läggas till.
**************************************************************************************************/
int double_vector_push(struct double_vector* self,
                       const double new_element)
{
   double* copy = (double*)realloc(self->data, sizeof(double) * (self->size + 1));
   if (!copy) return 1;
   copy[self->size++] = new_element;
   self->data = copy;
   return 0;
}

/**************************************************************************************************
* double_vector_pop: Tar bort ett element längst bak i angiven vektor, om ett sådant finns.
*
*                    - self: Pekare till vektorn.
**************************************************************************************************/
int double_vector_pop(struct double_vector* self)
{
   if (self->size <= 1)
   {
      double_vector_delete(self);
      return 1;
   }
   else
   {
      double* copy = (double*)realloc(self->data, sizeof(double) * (self->size - 1));
      if (!copy) return 1;
      self->data = copy;
      self->size--;
      return 0;
   }
}

/**************************************************************************************************
* double_vector_print: Skriver ut innehåll lagrat i angiven vektor via angiven utström, där
*                      standardutenheten stdout används som default för utskrift i terminalen.
*
*                      - self   : Pekare till vektorn.
*                      - ostream: Pekare till angiven utström (default = stdout).
**************************************************************************************************/
void double_vector_print(const struct double_vector* self,
                         FILE* ostream)
{
   if (!self->size) return;
   if (!ostream) ostream = stdout;
   fprintf(ostream, "--------------------------------------------------------------------------\n");

   for (const double* i = self->data; i < self->data + self->size; ++i)
   {
      fprintf(ostream, "%g\n", *i);
   }

   fprintf(ostream, "--------------------------------------------------------------------------\n\n");
   return;
}

/**************************************************************************************************
* double_vector_begin: Returnerar adressen till det första elementet i angiven vektor.
*
*                      - self: Pekare till vektorn.
**************************************************************************************************/
double* double_vector_begin(const struct double_vector* self)
{
   return self->data;
}

/**************************************************************************************************
* double_vector_end: Returnerar adressen direkt efter det sista elementet i angiven vektor.
*
*                    - self: Pekare till vektorn.
**************************************************************************************************/
double* double_vector_end(const struct double_vector* self)
{
   return self->data + self->size;
}

/**************************************************************************************************
* double_vector_clear: Tömmer innehållet i angiven vektor.
*
*                      - self: Pekare till vektorn.
**************************************************************************************************/
void (*double_vector_clear)(struct double_vector* self) = &double_vector_delete;


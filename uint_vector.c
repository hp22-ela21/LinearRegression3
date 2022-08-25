/**************************************************************************************************
* uint_vector.c: Innehåller funktionsdefinitioner för implementering av dynamiska vektorer
*                  för lagring av osignerade heltal via strukten uint_vector.
**************************************************************************************************/
#include "uint_vector.h"

/**************************************************************************************************
* uint_vector_new: Initierar angiven vektor.
* 
*                  - self: Pekare till vektorn.
**************************************************************************************************/
void uint_vector_new(struct uint_vector* self)
{
   self->data = 0;
   self->size = 0;
   return;
}

/**************************************************************************************************
* uint_vector_delete: Tömmer innehållet i angiven vektor.
* 
*                     - self: Pekare till vektorn.
**************************************************************************************************/
void uint_vector_delete(struct uint_vector* self)
{
   free(self->data);
   self->data = 0;
   self->size = 0;
   return;
}

/**************************************************************************************************
* uint_vector_ptr_new: Returnerar en pekare till en ny heapallokerad vektor av angiven storlek.
* 
*                      - size: Fältets storlek (antalet element det rymmer) vid start.
**************************************************************************************************/
struct uint_vector* uint_vector_ptr_new(const size_t size)
{
   struct uint_vector* self = (struct uint_vector*)malloc(sizeof(struct uint_vector));
   if (!self) return 0;
   self->data = 0;
   self->size = 0;
   uint_vector_resize(self, size);
   return self;
}

/**************************************************************************************************
* uint_vector_ptr_delete: Frigör minne för angiven heapallokerad vektor. Vektorpekarens adress
*                         passeras för att både frigöra minnet för det dynamiska fält denna pekar
*                         på, minnet för själva vektorn samt att vektorpekaren sätts till null.
* 
*                         - self: Adressen till vektorpekaren. 
**************************************************************************************************/
void uint_vector_ptr_delete(struct uint_vector** self)
{
   uint_vector_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* uint_vector_resize: Ändrar storleken / kapaciteten på angiven vektor via omallokering.
* 
*                     - self    : Pekare till vektorn.
*                     - new_size: Vektorns nya storlek efter omallokeringen.
**************************************************************************************************/
int uint_vector_resize(struct uint_vector* self, 
                       const size_t new_size)
{
   size_t* copy = (size_t*)realloc(self->data, sizeof(size_t) * new_size);
   if (!copy) return 1;
   self->data = copy;
   self->size = new_size;
   return 0;
}

/**************************************************************************************************
* uint_vector_push: Lägger till ett nytt element längst bak i angiven vektor.
* 
*                   - self       : Pekare till vektorn.
*                   - new_element: Det nya element som skall läggas till.
**************************************************************************************************/
int uint_vector_push(struct uint_vector* self, 
                     const size_t new_element)
{
   size_t* copy = (size_t*)realloc(self->data, sizeof(size_t) * (self->size + 1));
   if (!copy) return 1;
   copy[self->size++] = new_element;
   self->data = copy;
   return 0;
}

/**************************************************************************************************
* uint_vector_pop: Tar bort ett element längst bak i angiven vektor, om ett sådant finns.
* 
*                  - self: Pekare till vektorn.
**************************************************************************************************/
int uint_vector_pop(struct uint_vector* self)
{
   if (self->size <= 1)
   {
      uint_vector_delete(self);
      return 1;
   }
   else
   {
      size_t* copy = (size_t*)realloc(self->data, sizeof(size_t) * (self->size - 1));
      if (!copy) return 1;
      self->data = copy;
      self->size--;
      return 0;
   }
}

/**************************************************************************************************
* uint_vector_print: Skriver ut innehåll lagrat i angiven vektor via angiven utström, där
*                    standardutenheten stdout används som default för utskrift i terminalen.
* 
*                    - self   : Pekare till vektorn.
*                    - ostream: Pekare till angiven utström (default = stdout).
**************************************************************************************************/
void uint_vector_print(const struct uint_vector* self, 
                       FILE* ostream)
{
   if (!self->size) return;
   if (!ostream) ostream = stdout;
   fprintf(ostream, "--------------------------------------------------------------------------\n");

   for (const size_t* i = self->data; i < self->data + self->size; ++i)
   {
      fprintf(ostream, "%zu\n", *i);
   }

   fprintf(ostream, "--------------------------------------------------------------------------\n\n");
   return;
}

/**************************************************************************************************
* uint_vector_begin: Returnerar adressen till det första elementet i angiven vektor.
* 
*                    - self: Pekare till vektorn.
**************************************************************************************************/
size_t* uint_vector_begin(const struct uint_vector* self) 
{ 
   return self->data; 
}

/**************************************************************************************************
* uint_vector_end: Returnerar adressen direkt efter det sista elementet i angiven vektor.
* 
*                  - self: Pekare till vektorn.
**************************************************************************************************/
size_t* uint_vector_end(const struct uint_vector* self) 
{ 
   return self->data + self->size; 
}

/**************************************************************************************************
* uint_vector_clear: Tömmer innehållet i angiven vektor.
* 
*                    - self: Pekare till vektorn.
**************************************************************************************************/
void (*uint_vector_clear)(struct uint_vector* self) = &uint_vector_delete;



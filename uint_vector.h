/**************************************************************************************************
* uint_vector.h: Implementering av dynamiska vektorer för lagring av osignerade heltal via
*                strukten uint_vector samt motsvarande externa funktioner. 
**************************************************************************************************/
#ifndef UINT_VECTOR_H_
#define UINT_VECTOR_H_

/* Inkluderingsdirektiv: */
#include <stdio.h>
#include <stdlib.h>

/**************************************************************************************************
* uint_vector: Vektor innehållande ett dynamiskt fält för lagring av osignerade heltal. Antalet 
*              element som lagras i fältet räknas upp och uttrycks i form av vektorns storlek.
**************************************************************************************************/
struct uint_vector
{
   size_t* data; /* Pekare till dynamiskt fält för lagring av osignerade heltal. */
   size_t size;  /* Vektorns storlek (antalet element i fältet). */
};

/* Externa funktioner: */
void uint_vector_new(struct uint_vector* self);
void uint_vector_delete(struct uint_vector* self);
struct uint_vector* uint_vector_ptr_new(const size_t size);
void uint_vector_ptr_delete(struct uint_vector** self);
int uint_vector_resize(struct uint_vector* self, 
                       const size_t new_size);
int  uint_vector_push(struct uint_vector* self, 
                      const size_t new_element);
int uint_vector_pop(struct uint_vector* self);
void uint_vector_print(const struct uint_vector* self, 
                       FILE* ostream);
size_t* uint_vector_begin(const struct uint_vector* self);
size_t* uint_vector_end(const struct uint_vector* self);

/* Funktionspekare: */
extern void (*uint_vector_clear)(struct uint_vector* self);

#endif /* UINT_VECTOR_H_ */



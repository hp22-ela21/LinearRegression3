/**************************************************************************************************
* double_vector.h: Implementering av dynamiska vektorer för lagring av flyttal via strukten
*                  double_vector samt motsvarande externa funktioner. 
**************************************************************************************************/
#ifndef DOUBLE_VECTOR_H_
#define DOUBLE_VECTOR_H_

/* Inkluderingsdirektiv: */
#include <stdio.h>
#include <stdlib.h>

/**************************************************************************************************
* double_vector: Vektor innehållande ett dynamiskt fält för lagring av flyttal. Antalet element
*                som lagras i fältet räknas upp och uttrycks i form av vektorns storlek.
**************************************************************************************************/
struct double_vector
{
   double* data; /* Pekare till dynamiskt fält för lagring av flyttal. */
   size_t size;  /* Vektorns storlek (antalet element i fältet). */
};

/* Externa funktioner: */
void double_vector_new(struct double_vector* self);
void double_vector_delete(struct double_vector* self);
struct double_vector* double_vector_ptr_new(const size_t size);
void double_vector_ptr_delete(struct double_vector** self);
int double_vector_resize(struct double_vector* self,
                         const size_t new_size);
int double_vector_push(struct double_vector* self,
                       const double new_element);
int double_vector_pop(struct double_vector* self);
void double_vector_print(const struct double_vector* self,
                         FILE* ostream);
double* double_vector_begin(const struct double_vector* self);
double* double_vector_end(const struct double_vector* self);

/* Funktionspekare: */
extern void (*double_vector_clear)(struct double_vector* self);

#endif /* DOUBLE_VECTOR_H_ */

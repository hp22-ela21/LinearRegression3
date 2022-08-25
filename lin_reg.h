/**************************************************************************************************
* lin_reg.h: Inneh�ller funktionalitet f�r implementering av maskininl�rningsmodeller baserade p�
*            linj�r regression via strukten lin_reg samt externa funktioner.
**************************************************************************************************/
#ifndef LIN_REG_H_
#define LIN_REG_H_

/* Inkluderingsdirektiv: */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "double_vector.h"
#include "uint_vector.h"

/**************************************************************************************************
* lin_reg: Strukt f�r implementering av maskininl�rningsmodeller baserade p� linj�r regression. 
*          Tr�ningsdata best�ende av valfritt antal tr�ningsupps�ttningar kan l�sas in fr�n en 
*          fil eller passeras via pekare till arrayer.
**************************************************************************************************/
struct lin_reg
{
   struct double_vector train_in;  /* Tr�ningsupps�ttningarnas insignaler. */
   struct double_vector train_out; /* Tr�ningsupps�ttningarnas utsignaler. */
   struct uint_vector train_order; /* Lagrar tr�ningsupps�ttningarnas ordningsf�ljd. */
   double bias;                    /* Vilov�rde (m-v�rde). */
   double weight;                  /* Lutning (k-v�rde). */
};

/* Externa funktioner: */
void lin_reg_new(struct lin_reg* self);
void lin_reg_delete(struct lin_reg* self);
struct lin_reg* lin_reg_ptr_new(void);
void lin_reg_ptr_delete(struct lin_reg** self);
void lin_reg_load_training_data(struct lin_reg* self, 
                                const char* filepath);
void lin_reg_set_training_data(struct lin_reg* self,
                               const double* train_in, 
                               const double* train_out, 
                               const size_t num_sets);
void lin_reg_train(struct lin_reg* self,
                   const size_t num_epochs,
                   const double learning_rate);
double lin_reg_predict(const struct lin_reg* self, 
                       const double input);
void lin_reg_predict_all(const struct lin_reg* self,
                         const double threshold, 
                         FILE* ostream);
void lin_reg_predict_range(const struct lin_reg* self, 
                           const double start_val,
                           const double end_val, 
                           const double step, 
                           const double threshold, 
                           FILE* ostream);

#endif /* LIN_REG_H_ */
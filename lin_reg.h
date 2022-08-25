/**************************************************************************************************
* lin_reg.h: Innehåller funktionalitet för implementering av maskininlärningsmodeller baserade på
*            linjär regression via strukten lin_reg samt externa funktioner.
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
* lin_reg: Strukt för implementering av maskininlärningsmodeller baserade på linjär regression. 
*          Träningsdata bestående av valfritt antal träningsuppsättningar kan läsas in från en 
*          fil eller passeras via pekare till arrayer.
**************************************************************************************************/
struct lin_reg
{
   struct double_vector train_in;  /* Träningsuppsättningarnas insignaler. */
   struct double_vector train_out; /* Träningsuppsättningarnas utsignaler. */
   struct uint_vector train_order; /* Lagrar träningsuppsättningarnas ordningsföljd. */
   double bias;                    /* Vilovärde (m-värde). */
   double weight;                  /* Lutning (k-värde). */
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
/**************************************************************************************************
* lin_reg.c: Innehåller externa funktioner avsedda för strukten lin_reg, som används för
*            implementering av maskininlärningsmodeller som baseras på linjär regression.
**************************************************************************************************/
#include "lin_reg.h"

// Statiska funktioner:
static void lin_reg_shuffle(struct lin_reg* self);
static void lin_reg_optimize(struct lin_reg* self,
                             const double input, 
                             const double reference,
                             const double learning_rate);
static void lin_reg_extract(struct lin_reg* self, 
                            const char* s);
static bool char_is_digit(const char c);
static void retrieve_double(struct double_vector* data, 
                            char* s);

/**************************************************************************************************
* lin_reg_new: Initierar angiven regressionsmodell. Träningsdata måste tillföras i efterhand via 
*              någon av funktioner lin_reg_load_training_data (för inläsning av träningsdata 
*              från en textfil) eller lin_reg_set_training_data (för att passera pekare till 
*              arrayer innehållande träningsdata).
* 
*              - self         : Pekare till regressionsmodellen.
**************************************************************************************************/
void lin_reg_new(struct lin_reg* self)
{
   double_vector_new(&self->train_in);
   double_vector_new(&self->train_out);
   uint_vector_new(&self->train_order);
   self->bias = 0;
   self->weight = 0;
   return;
}

/**************************************************************************************************
* lin_reg_delete: Nollställer angiven regressionsmodell. Minnet för modellen frigörs dock inte, 
*                 så denna kan återanvändas vid behov.
* 
*                 - self: Pekare till regressionsmodellen.
**************************************************************************************************/
void lin_reg_delete(struct lin_reg* self)
{
   double_vector_delete(&self->train_in);
   double_vector_delete(&self->train_out);
   uint_vector_delete(&self->train_order);
   self->bias = 0;
   self->weight = 0;
   return;
}

/**************************************************************************************************
* lin_reg_ptr_new: Returnerar en pekare till en ny heapallokerad regressionsmodell. Träningsdata 
*                  måste tillföras i efterhand via någon av funktioner lin_reg_load_training_data 
*                  (för inläsning av träningsdata från en textfil) eller lin_reg_set_training_data 
*                  (för att passera pekare till arrayer innehållande träningsdata).
**************************************************************************************************/
struct lin_reg* lin_reg_ptr_new(void)
{
   struct lin_reg* self = (struct lin_reg*)malloc(sizeof(struct lin_reg));
   if (!self) return 0;
   lin_reg_new(self);
   return self;
}

/**************************************************************************************************
* lin_reg_delete: Nollställer och frigör minne för angiven heapallokerad regressionsmodell. 
*                 Pekaren till regressionsmodellen sätts till null efter att minnet har frigjorts.
* 
*                 - self: Adressen till regressionsmodellpekaren.
**************************************************************************************************/
void lin_reg_ptr_delete(struct lin_reg** self)
{
   lin_reg_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/**************************************************************************************************
* lin_reg_load_training_data: Läser in träningsdata till angiven regressionsmodell från en fil
*                             via angiven filsökväg.
* 
*                             - self    : Pekare till regressionsmodellen.
*                             - filepath: Pekare till filsökvägen.
**************************************************************************************************/
void lin_reg_load_training_data(struct lin_reg* self, 
                                const char* filepath)
{
   FILE* fstream = fopen(filepath, "r");

   if (!fstream)
   {
      fprintf(stderr, "Could not open file at path %s!\n\n", filepath);
   }
   else
   {
      char s[100] = { '\0' };
      while (fgets(s, (int)sizeof(s), fstream))
      {
         lin_reg_extract(self, s);
      }
      fclose(fstream);
   }

   return;
}

/**************************************************************************************************
* lin_reg_set_training_data: Kopierar träningsdata till angiven regressionsmodell från refererade
*                            arrayer samt lagrar index för respektive träningsuppsättning.
* 
*                            - self     : Pekare till regressionsmodellen.
*                            - train_in : Pekare till array innehållande insignaler.
*                            - train_out: Pekare till array innehållande referensvärden.
*                            - num_sets : Antalet passerade träningsuppsättningar.
**************************************************************************************************/
void lin_reg_set_training_data(struct lin_reg* self,
                               const double* train_in, 
                               const double* train_out, 
                               const size_t num_sets)
{
   const size_t new_size = self->train_in.size + num_sets;
   const size_t offset = self->train_in.size;

   double_vector_resize(&self->train_in, new_size);
   double_vector_resize(&self->train_out, new_size);
   uint_vector_resize(&self->train_order, new_size);

   for (size_t i = 0; i < num_sets; ++i)
   {
      self->train_in.data[offset + i] = train_in[i];
      self->train_out.data[offset + i] = train_out[i];
      self->train_order.data[offset + i] = offset + i;
   }

   return;
}

/**************************************************************************************************
* lin_reg_train: Tränar angiven regressionsmodell med givet antal epoker samt given lärhastighet. 
*                I början av varje epok randomiseras ordningsföljden på träningsuppsättningarna 
*                för att undvika att eventuella mönster som förekommer i träningsdatan skall 
*                påverka träningen.
* 
*                - self         : Pekare till regressionsmodellen.
*                - num_epochs   : Antalet epoker som skall genomföras vid träning.
*                - learning_rate: Den lärhastighet som skall användas vid träning för att
*                                 justera modellens parametrar vid avvikelse.
**************************************************************************************************/
void lin_reg_train(struct lin_reg* self,
                   const size_t num_epochs,
                   const double learning_rate)
{
   for (size_t i = 0; i < num_epochs; ++i)
   {
      lin_reg_shuffle(self);

      for (size_t j = 0; j < self->train_order.size; ++j)
      {
         const size_t k = self->train_order.data[j];
         lin_reg_optimize(self, self->train_in.data[k], self->train_out.data[k], learning_rate);
      }
   }

   return;
}

/**************************************************************************************************
* lin_reg_predict: Genomför prediktion med angiven regressionsmodell via angiven insignal och
*                  returnerar det predikterade resultatet.
* 
*                  - self : Pekare till regressionsmodellen.
*                  - input: Insignal som skall användas för prediktion.
**************************************************************************************************/
double lin_reg_predict(const struct lin_reg* self, 
                       const double input)
{
   return self->weight * input + self->bias;
}

/**************************************************************************************************
* lin_reg_predict_all: Genomför prediktion med angiven regressionsmodell för samtliga insignaler 
*                      från träningsdatan och skriver ut motsvarande predikterade utsignaler via 
*                      angiven utström, där standardutenheten stdout används som default för 
*                      utskrift i terminalen. Värden mycket nära noll avrundas för att undvika 
*                      utskrift med ett flertal decimaler.
* 
*                      - self     : Pekare till regressionsmodellen.
*                      - threshold: Tröskelvärde, där samtliga predikterade värden som ligger 
*                                   inom intervallet [-threshold, threshold] avrundas till noll.
*                      - ostream  : Pekare till angiven utström (default = stdout).
**************************************************************************************************/
void lin_reg_predict_all(const struct lin_reg* self,
                         const double threshold, 
                         FILE* ostream)
{
   if (!ostream) ostream = stdout;
   if (!self->train_in.size) return;

   const size_t last = self->train_in.size - 1;
   fprintf(ostream, "--------------------------------------------------------------------------\n");

   for (size_t i = 0; i < self->train_in.size; ++i)
   {
      const double prediction = self->weight * self->train_in.data[i] + self->bias;
      fprintf(ostream, "Input: %g", self->train_in.data[i]);

      if (prediction < threshold && prediction > -threshold)
      {
         fprintf(ostream, "Output: %g", 0.0);
      }
      else
      {
         fprintf(ostream, "Output: %g", prediction);
      }

      if (i < last) fprintf(ostream, "\n");
   }

   fprintf(ostream, "--------------------------------------------------------------------------\n\n");
   return;
}

/**************************************************************************************************
* lin_reg_predict_range: Genomför prediktion med angiven regressionsmodell för insignaler mellan 
*                        angivet start- och slutvärde i steg om angiven stegvärde. Motsvarande 
*                        predikterad utsignal skrivs ut via angiven utström. Värden mycket nära 
*                        noll avrundas för att undvika utskrift med ett flertal decimaler.
* 
*                        - self     : Pekare till regressionsmodellen.
*                        - start_val: Minvärde för insignaler som skall testas.
*                        - end_val  : Maxvärde för insignaler som skall testas.
*                        - step     : Stegvärde/inkrementeringsvärde för insignaler.
*                        - threshold: Tröskelvärde, där samtliga predikterade värden inom
*                                     intervallet [-threshold, threshold] avrundas till noll.
*                         - ostream : Pekare till angiven utström (default = stdout).
**************************************************************************************************/
void lin_reg_predict_range(const struct lin_reg* self, 
                           const double start_val,
                           const double end_val, 
                           const double step, 
                           const double threshold, 
                           FILE* ostream)
{
   if (!self->train_in.size) return;
   if (!ostream) ostream = stdout;
   fprintf(ostream, "--------------------------------------------------------------------------\n");

   for (double i = start_val; i <= end_val; i += step)
   {
      const double prediction = self->weight * i + self->bias;
      fprintf(ostream, "Input: %g\n", i);

      if (prediction < threshold && prediction > -threshold)
      {
         fprintf(ostream, "Output: %g\n", 0.0);
      }
      else
      {
         fprintf(ostream, "Output: %g\n", prediction);
      }

      if (i < end_val) fprintf(ostream, "\n");
   }

   fprintf(ostream, "--------------------------------------------------------------------------\n\n");
   return;
}

/**************************************************************************************************
* lin_reg_shuffle: Randomiserar den inbördes ordningsföljden för angiven regressionsmodells 
*                  träningsuppsättningar.
* 
*                  - self: Pekare till regressionsmodellen.
**************************************************************************************************/
static void lin_reg_shuffle(struct lin_reg* self)
{
   for (size_t i = 0; i < self->train_order.size; ++i)
   {
      const size_t r = (size_t)rand() % self->train_order.size;
      const size_t temp = self->train_order.data[i];
      self->train_order.data[i] = self->train_order.data[r];
      self->train_order.data[r] = temp;
   }
   return;
}

/**************************************************************************************************
* lin_reg_optimize: Justerar parametrar för angiven regressionsmodell med målsättningen att minska 
*                   aktuell avvikelse. Prediktion genomförs via angiven insignal, där predikterad 
*                   utdat jämförs mot givet referensvärde för att beräkna aktuell avvikelse, som 
*                   tillsammans med lärhastigheten avgör graden av justering.
*
*                   - self         : Pekare till regressionsmodellen.
*                   - input        : Insignal från träningsdata, som används för prediktion.
*                   - reference    : Referensvärde från träningsdata, som jämförs mot predikterad
*                                    utsignal för att beräkna aktuellt fel.
*                   - learning_rate: Den lärhastighet som skall användas vid träning för att
*                                    justera modellens parametrar vid avvikelse.
**************************************************************************************************/
static void lin_reg_optimize(struct lin_reg* self,
                             const double input, 
                             const double reference,
                             const double learning_rate)
{
   const double prediction = self->weight * input + self->bias;
   const double error = reference - prediction;
   const double change_rate = error * learning_rate;
   self->bias += change_rate;
   self->weight += change_rate * input;
   return;
}

/**************************************************************************************************
* lin_reg_extract: Extraherar träningsdata i form av flyttal ur angivet textstycke. Ifall två 
*                  flyttal lyckas extraheras så lagras dessa som en träningsuppsättning. Index 
*                  för träningsuppsättningen lagras också för att enkelt kunna randomisera 
*                  uppsättningarnas ordningsföljd vid träning utan att förflytta träningsdatan.
* 
*                  - self: Pekare till regressionsmodellen.
*                  - s   : Pekare till textstycket som flyttal extraheras ur.
**************************************************************************************************/
static void lin_reg_extract(struct lin_reg* self, 
                            const char* s)
{
   char num_str[20] = { '\0 ' };
   size_t index = 0;
   struct double_vector numbers = { .data = 0, .size = 0 };

   for (const char* i = s; *i; ++i)
   {
      if (char_is_digit(*i))
      {
         num_str[index++] = *i;
      }
      else
      {
         retrieve_double(&numbers, num_str);
         index = 0;
      }
   }

   if (index)
   {
      retrieve_double(&numbers, num_str);
   }

   if (numbers.size == 2)
   {
      double_vector_push(&self->train_in, numbers.data[0]);
      double_vector_push(&self->train_out, numbers.data[1]);
      uint_vector_push(&self->train_order, self->train_order.size);
   }

   double_vector_delete(&numbers);
   return;
}

/**************************************************************************************************
* char_is_digit: Indikerar ifall givet tecken utgör en siffra eller ett relaterat tecken, såsom
*                ett minustecken eller en punkt. Eftersom flyttal ibland matas in både med
*                punkt samt kommatecken så utgör båda giltiga tecken.
* 
*                - c: Det tecken som skall kontrolleras.
**************************************************************************************************/
static bool char_is_digit(const char c)
{
   const char* s = "0123456789-.,";

   for (const char* i = s; *i; ++i)
   {
      if (*i == c) return true;
   }

   return false;
}

/**************************************************************************************************
* retrieve_double: Typomvandlar innehåll lagrat som text till ett flyttal och lagrar resultatet
*                  i en vektor. Innan typomvandlingen äger rum ersätts eventuella kommatecken
*                  med punkt, vilket möjliggör att flyttal kan läsas in både med punkt eller
*                  kommatecken som decimaltecken.
* 
*                  - data: Pekare till den vektor som typomvandlat flyttal skall lagras i.
*                  - s   : Pekare till det textstycke som skall typomvandlas till ett flyttal.
**************************************************************************************************/
static void retrieve_double(struct double_vector* data, 
                            char* s)
{
   for (char* i = s; *i; ++i)
   {
      if (*i == ',') *i = '.';
   }

   const double num = atof(s);
   double_vector_push(data, num);
   s[0] = '\0';
   return;
}
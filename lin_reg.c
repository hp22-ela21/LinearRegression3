/**************************************************************************************************
* lin_reg.c: Inneh�ller externa funktioner avsedda f�r strukten lin_reg, som anv�nds f�r
*            implementering av maskininl�rningsmodeller som baseras p� linj�r regression.
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
* lin_reg_new: Initierar angiven regressionsmodell. Tr�ningsdata m�ste tillf�ras i efterhand via 
*              n�gon av funktioner lin_reg_load_training_data (f�r inl�sning av tr�ningsdata 
*              fr�n en textfil) eller lin_reg_set_training_data (f�r att passera pekare till 
*              arrayer inneh�llande tr�ningsdata).
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
* lin_reg_delete: Nollst�ller angiven regressionsmodell. Minnet f�r modellen frig�rs dock inte, 
*                 s� denna kan �teranv�ndas vid behov.
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
* lin_reg_ptr_new: Returnerar en pekare till en ny heapallokerad regressionsmodell. Tr�ningsdata 
*                  m�ste tillf�ras i efterhand via n�gon av funktioner lin_reg_load_training_data 
*                  (f�r inl�sning av tr�ningsdata fr�n en textfil) eller lin_reg_set_training_data 
*                  (f�r att passera pekare till arrayer inneh�llande tr�ningsdata).
**************************************************************************************************/
struct lin_reg* lin_reg_ptr_new(void)
{
   struct lin_reg* self = (struct lin_reg*)malloc(sizeof(struct lin_reg));
   if (!self) return 0;
   lin_reg_new(self);
   return self;
}

/**************************************************************************************************
* lin_reg_delete: Nollst�ller och frig�r minne f�r angiven heapallokerad regressionsmodell. 
*                 Pekaren till regressionsmodellen s�tts till null efter att minnet har frigjorts.
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
* lin_reg_load_training_data: L�ser in tr�ningsdata till angiven regressionsmodell fr�n en fil
*                             via angiven fils�kv�g.
* 
*                             - self    : Pekare till regressionsmodellen.
*                             - filepath: Pekare till fils�kv�gen.
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
* lin_reg_set_training_data: Kopierar tr�ningsdata till angiven regressionsmodell fr�n refererade
*                            arrayer samt lagrar index f�r respektive tr�ningsupps�ttning.
* 
*                            - self     : Pekare till regressionsmodellen.
*                            - train_in : Pekare till array inneh�llande insignaler.
*                            - train_out: Pekare till array inneh�llande referensv�rden.
*                            - num_sets : Antalet passerade tr�ningsupps�ttningar.
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
* lin_reg_train: Tr�nar angiven regressionsmodell med givet antal epoker samt given l�rhastighet. 
*                I b�rjan av varje epok randomiseras ordningsf�ljden p� tr�ningsupps�ttningarna 
*                f�r att undvika att eventuella m�nster som f�rekommer i tr�ningsdatan skall 
*                p�verka tr�ningen.
* 
*                - self         : Pekare till regressionsmodellen.
*                - num_epochs   : Antalet epoker som skall genomf�ras vid tr�ning.
*                - learning_rate: Den l�rhastighet som skall anv�ndas vid tr�ning f�r att
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
* lin_reg_predict: Genomf�r prediktion med angiven regressionsmodell via angiven insignal och
*                  returnerar det predikterade resultatet.
* 
*                  - self : Pekare till regressionsmodellen.
*                  - input: Insignal som skall anv�ndas f�r prediktion.
**************************************************************************************************/
double lin_reg_predict(const struct lin_reg* self, 
                       const double input)
{
   return self->weight * input + self->bias;
}

/**************************************************************************************************
* lin_reg_predict_all: Genomf�r prediktion med angiven regressionsmodell f�r samtliga insignaler 
*                      fr�n tr�ningsdatan och skriver ut motsvarande predikterade utsignaler via 
*                      angiven utstr�m, d�r standardutenheten stdout anv�nds som default f�r 
*                      utskrift i terminalen. V�rden mycket n�ra noll avrundas f�r att undvika 
*                      utskrift med ett flertal decimaler.
* 
*                      - self     : Pekare till regressionsmodellen.
*                      - threshold: Tr�skelv�rde, d�r samtliga predikterade v�rden som ligger 
*                                   inom intervallet [-threshold, threshold] avrundas till noll.
*                      - ostream  : Pekare till angiven utstr�m (default = stdout).
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
* lin_reg_predict_range: Genomf�r prediktion med angiven regressionsmodell f�r insignaler mellan 
*                        angivet start- och slutv�rde i steg om angiven stegv�rde. Motsvarande 
*                        predikterad utsignal skrivs ut via angiven utstr�m. V�rden mycket n�ra 
*                        noll avrundas f�r att undvika utskrift med ett flertal decimaler.
* 
*                        - self     : Pekare till regressionsmodellen.
*                        - start_val: Minv�rde f�r insignaler som skall testas.
*                        - end_val  : Maxv�rde f�r insignaler som skall testas.
*                        - step     : Stegv�rde/inkrementeringsv�rde f�r insignaler.
*                        - threshold: Tr�skelv�rde, d�r samtliga predikterade v�rden inom
*                                     intervallet [-threshold, threshold] avrundas till noll.
*                         - ostream : Pekare till angiven utstr�m (default = stdout).
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
* lin_reg_shuffle: Randomiserar den inb�rdes ordningsf�ljden f�r angiven regressionsmodells 
*                  tr�ningsupps�ttningar.
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
* lin_reg_optimize: Justerar parametrar f�r angiven regressionsmodell med m�ls�ttningen att minska 
*                   aktuell avvikelse. Prediktion genomf�rs via angiven insignal, d�r predikterad 
*                   utdat j�mf�rs mot givet referensv�rde f�r att ber�kna aktuell avvikelse, som 
*                   tillsammans med l�rhastigheten avg�r graden av justering.
*
*                   - self         : Pekare till regressionsmodellen.
*                   - input        : Insignal fr�n tr�ningsdata, som anv�nds f�r prediktion.
*                   - reference    : Referensv�rde fr�n tr�ningsdata, som j�mf�rs mot predikterad
*                                    utsignal f�r att ber�kna aktuellt fel.
*                   - learning_rate: Den l�rhastighet som skall anv�ndas vid tr�ning f�r att
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
* lin_reg_extract: Extraherar tr�ningsdata i form av flyttal ur angivet textstycke. Ifall tv� 
*                  flyttal lyckas extraheras s� lagras dessa som en tr�ningsupps�ttning. Index 
*                  f�r tr�ningsupps�ttningen lagras ocks� f�r att enkelt kunna randomisera 
*                  upps�ttningarnas ordningsf�ljd vid tr�ning utan att f�rflytta tr�ningsdatan.
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
* char_is_digit: Indikerar ifall givet tecken utg�r en siffra eller ett relaterat tecken, s�som
*                ett minustecken eller en punkt. Eftersom flyttal ibland matas in b�de med
*                punkt samt kommatecken s� utg�r b�da giltiga tecken.
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
* retrieve_double: Typomvandlar inneh�ll lagrat som text till ett flyttal och lagrar resultatet
*                  i en vektor. Innan typomvandlingen �ger rum ers�tts eventuella kommatecken
*                  med punkt, vilket m�jligg�r att flyttal kan l�sas in b�de med punkt eller
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
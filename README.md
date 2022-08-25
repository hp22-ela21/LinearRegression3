# LinearRegression3
Implementering av en maskininlärningsmodell som baseras på linjär regression i C.

Detta exempel kan tänkas utgöra en C-version av exemplet LinearRegression2, där programkoden skrevs i C++.
I detta fall används en strukt döpt lin_reg samt ett flertal externa funktioner för att realisera modellen. 
Dynamiska vektorer för flyttal samt osignerade tal implementeras via struktar double_vector samt uint_vector, 
tillsammans med ett flertal externa funktioner.

I detta exempel används tio träningsuppsättningar definierade enligt formeln y = -5x + 0.5, som läses in från en textfil döpt data.txt. 
Träning sker under 1000 epoker med en lärhastighet på 1 %, följt av att modellen testas, i detta fall för samtliga flyttal mellan -10 och 10 i intervall om 1. 
Resultatet skrivs ut i terminalen och indikerar att modellen predikterar med 100 % precision.

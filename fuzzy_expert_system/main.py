import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

#2

# Zbiór danych
smak = ctrl.Antecedent(np.arange(0, 11, 1), 'smak')
pikantnosc = ctrl.Antecedent(np.arange(0, 11, 1), 'pikantnosc')
konsystencja = ctrl.Antecedent(np.arange(0, 11, 1), 'konsystencja')
slodycz = ctrl.Antecedent(np.arange(0, 11, 1), 'slodycz')
przydatnosc = ctrl.Consequent(np.arange(0, 11, 1), 'przydatnosc')

# Definicja funkcji przynależności rozmytej dla cech potraw
smak['poor'] = fuzz.trimf(smak.universe, [0, 0, 5])
smak['average'] = fuzz.trimf(smak.universe, [0, 5, 10])
smak['good'] = fuzz.trimf(smak.universe, [5, 10, 10])

pikantnosc['poor'] = fuzz.trimf(pikantnosc.universe, [0, 0, 5])
pikantnosc['average'] = fuzz.trimf(pikantnosc.universe, [0, 5, 10])
pikantnosc['good'] = fuzz.trimf(pikantnosc.universe, [5, 10, 10])

konsystencja['poor'] = fuzz.trimf(konsystencja.universe, [0, 0, 5])
konsystencja['average'] = fuzz.trimf(konsystencja.universe, [0, 5, 10])
konsystencja['good'] = fuzz.trimf(konsystencja.universe, [5, 10, 10])

slodycz['poor'] = fuzz.trimf(slodycz.universe, [0, 0, 5])
slodycz['average'] = fuzz.trimf(slodycz.universe, [0, 5, 10])
slodycz['good'] = fuzz.trimf(slodycz.universe, [5, 10, 10])

przydatnosc['poor'] = fuzz.trimf(przydatnosc.universe, [0, 0, 5])
przydatnosc['average'] = fuzz.trimf(przydatnosc.universe, [0, 5, 10])
przydatnosc['good'] = fuzz.trimf(przydatnosc.universe, [5, 10, 10])

#1

# Zbiór reguł
regula1 = ctrl.Rule(smak['good'] & pikantnosc['average'], przydatnosc['good'])
regula2 = ctrl.Rule(smak['poor'] | konsystencja['poor'], przydatnosc['poor'])
regula3 = ctrl.Rule(slodycz['good'] & (pikantnosc['average'] | konsystencja['good']), przydatnosc['good'])
regula4 = ctrl.Rule(smak['average'] & pikantnosc['good'] & slodycz['average'], przydatnosc['average'])
regula5 = ctrl.Rule(smak['good'] & konsystencja['average'] & slodycz['good'], przydatnosc['good'])
regula6 = ctrl.Rule(smak['poor'] & pikantnosc['poor'] & konsystencja['poor'], przydatnosc['poor'])
regula7 = ctrl.Rule(smak['average'] & pikantnosc['poor'] & slodycz['poor'], przydatnosc['poor'])
regula8 = ctrl.Rule(smak['poor'] & pikantnosc['average'] & konsystencja['average'], przydatnosc['poor'])
regula9 = ctrl.Rule(smak['good'] & pikantnosc['good'] & slodycz['good'], przydatnosc['good'])
regula10 = ctrl.Rule(smak['average'] & konsystencja['poor'] & slodycz['average'], przydatnosc['poor'])

#3/#4

# Mechanizm wnioskowania rozmytego
system_ctrl = ctrl.ControlSystem([regula1, regula2, regula3, regula4, regula5, regula6, regula7, regula8, regula9, regula10])
ocena_potrawy = ctrl.ControlSystemSimulation(system_ctrl)

# Interfejs użytkownika
print("Podaj cechy potrawy (skala od 0 do 10):")
ocena_potrawy.input['smak'] = float(input("Smak: "))
ocena_potrawy.input['pikantnosc'] = float(input("Pikantność: "))
ocena_potrawy.input['konsystencja'] = float(input("Konsystencja: "))
ocena_potrawy.input['slodycz'] = float(input("Słodycz: "))

# Wykonanie wnioskowania
ocena_potrawy.compute()

# Wyświetlenie wyniku
print("\nOcena przydatności potrawy:", ocena_potrawy.output['przydatnosc'])

# Wykresy funkcji przynależności i wyniku wnioskowania
smak.view(sim=ocena_potrawy)
pikantnosc.view(sim=ocena_potrawy)
konsystencja.view(sim=ocena_potrawy)
slodycz.view(sim=ocena_potrawy)
przydatnosc.view(sim=ocena_potrawy)
plt.show()
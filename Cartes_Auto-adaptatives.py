# Importation des bibliothéques
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patches

''' *********************** Partie 1 : Initialisation *********************** '''

#100 rangées de vecteurs 3D toutes entre les valeurs de 0 et 255.
input_data = np.random.randint(0, 255, (3, 100))
#Une carte de taille 5 x 5
carte_dimensions = np.array([5, 5])
iterations = 4000
rate = 0.01
# établir des variables de taille en fonction des données
m = input_data.shape[0]
n = input_data.shape[1]
# le SOM doit être un vecteur à m dimensions pour chaque neurone
weights_vecteur = np.random.random((carte_dimensions[0], carte_dimensions[1], m))
# initialiser le rayon de voisinage
radius = max(carte_dimensions[0], carte_dimensions[1]) / 2
print(radius)
# paramètre de décroissance du rayon
time = iterations / np.log(radius)


''' *********************** Partie 2 : Normalisation *********************** '''

isNormalise = True

# si True, supposons que toutes les données sont à l'échelle commune
# si False, normaliser à [0 1] dans chaque colonne
isColumnNormalise = False

# nous voulons conserver une copie des données brutes pour plus tard
copie_data = input_data

# vérifier si les données doivent être normalisées
if isNormalise:
    if isColumnNormalise:
        # normaliser le long de chaque colonne
        maxColumn = input_data.max(axis=0)
        copie_data = input_data / maxColumn[np.newaxis, :]
    else:
        # normaliser l'ensemble de données
        copie_data = input_data / copie_data.max()


''' *********************** Partie 3 : Apprentissage (ou Learning) *********************** '''

def bestMatchingUnit(t, weights_vecteur, m):
    """
        Trouvez la meilleure unité correspondante pour un vecteur donné, t, dans
        le SOM renvoie: un tuple (bmu, bmuIndex) où bmu est le BMU de haute dimension et
        bmuIndex est l'indice de ce vecteur dans le SOM
    """
    bmuIndex = np.array([0, 0])
    # définir la distance minimale initiale à un nombre énorme
    distanceMin = np.iinfo(np.int).max
    # calculer la distance à haute dimension entre chaque neurone et l'entrée
    for x in range(weights_vecteur.shape[0]):
        for y in range(weights_vecteur.shape[1]):
            w = weights_vecteur[x, y, :].reshape(m, 1)
            # ne nous embêtons pas avec la distance euclidienne réelle, pour éviter une opération sqrt coûteuse
            distanceSqrt = np.sum((w - t) ** 2)
            if distanceSqrt < distanceMin:
                distanceMin = distanceSqrt
                bmuIndex = np.array([x, y])
            # obtenir le vecteur correspondant à bmuIndex
            bmu = weights_vecteur[bmuIndex[0], bmuIndex[1], :].reshape(m, 1)
            # retourner le tuple (bmu, bmuIndex)
    return (bmu, bmuIndex)



def diminuerRadius(initial_radius, it, time):
    return initial_radius * np.exp(-it / time)

def diminuerRate(initial_learning_rate, it, iterations):
    return initial_learning_rate * np.exp(-it / iterations)

def calculerInfluence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))


for it in range(iterations):
    # sélectionner un exemple de formation au hasard
    t = copie_data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))

    # trouver son unité la mieux adaptée
    bmu, bmuIndex = bestMatchingUnit(t, weights_vecteur, m)

    # diminuer les paramétres du SOM
    r = diminuerRadius(radius, it, time)
    l = diminuerRate(rate, it, iterations)

    # maintenant nous connaissons le BMU, mettons à jour son vecteur de poids pour nous rapprochons de l'entrée
    # et rapprochez ses voisins dans un espace 2D
    # par un facteur proportionnel à leur distance 2D par rapport au BMU

    ''' Adaptation des neurones '''
    for x in range(weights_vecteur.shape[0]):
        for y in range(weights_vecteur.shape[1]):
            w = weights_vecteur[x, y, :].reshape(m, 1)
            # obtenir la distance 2D (encore une fois, pas la distance euclidienne réelle)
            distanceWeight = np.sum((np.array([x, y]) - bmuIndex) ** 2)
            # si la distance est dans le rayon de voisinage actuel
            if distanceWeight <= r**2:
                # calculer le degré d'influence (basé sur la distance 2D)
                influence = calculerInfluence(distanceWeight, r)
                # calculer le degré d'influence (basé sur la distance 2D)
                # new w = old w + (learning rate * influence * delta)
                # where delta = input vector (t) - old w
                newWeight = w + (l * influence * (t - w))
                # enregistrer le nouveau poids
                weights_vecteur[x, y, :] = newWeight.reshape(1, 3)

''' *********************** Partie 4 : Visualisation *********************** '''

figure = plt.figure()
#axes
axes = figure.add_subplot(111, aspect='equal')
axes.set_xlim((0, weights_vecteur.shape[0]+1))
axes.set_ylim((0, weights_vecteur.shape[1]+1))
axes.set_title('La carte auto adaptative aprés %d itérations' % iterations)

# Dessiner les rectangles dans la figure
for x in range(1, weights_vecteur.shape[0] + 1):
    for y in range(1, weights_vecteur.shape[1] + 1):
        axes.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1,
                     facecolor=weights_vecteur[x-1,y-1,:],
                     edgecolor='none'))

#Afficher les résultats
plt.show()
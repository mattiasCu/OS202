from PIL import Image 
import numpy as np
from numpy import linalg
import scipy as sp
from scipy import sparse
import time
from mpi4py import MPI
import colorize

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

niters = 50_000
epsilon = 1.E-10

HUE       = 0
SATURATION= 1
INTENSITY = 2

Y         = 0
CB        = 1
CR        = 2



# On charge l'image en noir et blanc
gray_img = "example.bmp"
marked_img = "example_marked.bmp"
output = "example.png"

im_gray = Image.open(gray_img)
im_gray = im_gray.convert('HSV')
# On convertit l'image en tableau (ny x nx x 3) (Trois pour les trois composantes de la couleur)
values_gray = np.array(im_gray)

sendbuff = None
recvbuff = None
residu = False

height, width, channels = values_gray.shape

ligne_div=int(height/size)

nom_elements_par_processus=ligne_div*width*channels

recubuff=np.empty(nom_elements_par_processus,dtype=np.uint8)

if rank ==0:
    sendbuff = values_gray.flatten()

try:
    comm.Scatter(sendbuff, recubuff, root=0)
except:
    print("Erreur dans la scatter")
    exit()

values_gray = recubuff.reshape((ligne_div,width,channels))
intensity = (1./255.)*colorize.create_field(values_gray, INTENSITY, nb_layers=2, prolong_field=True)

# Calcul de la moyenne de l'intensite pour chaque pixel avec ses huit voisins
# La moyenne contient une couche de cellules fantomes (une de moins que l'intensite)
deb = time.time()
means = colorize.compute_means(intensity)
end = time.time() - deb
print(f"Temps calcul moyenne : {end} secondes")
# Calcul de la variance de l'intensite pour chaque pixel avec ses huit voisins
# La variance contient une couche de cellules fantomes comme la moyenne.
deb = time.time()
variance = colorize.compute_variance(intensity, means)
end = time.time() - deb
print(f"Temps calcul variance : {end} secondes")

# Calcul de la matrice utilisee pour minimiser la fonction quadratique
deb = time.time()
A = colorize.compute_matrix((means.shape[1]-2,means.shape[0]-2), 0, intensity, means, variance)
end = time.time() - deb
print(f"Temps calcul matrice : {end} secondes")

# Calcul des seconds membres
im = Image.open(marked_img)
im_ycbcr = im.convert('YCbCr')
val_ycbcr = np.array(im_ycbcr)

#=====================================
sendbuff = None
recvbuff = None
residu = False

height, width, channels = val_ycbcr.shape

ligne_div=int(height/size)

nom_elements_par_processus=ligne_div*width*channels

recubuff=np.empty(nom_elements_par_processus,dtype=np.uint8)

if rank ==0:
    sendbuff = val_ycbcr.flatten()

try:
    comm.Scatter(sendbuff, recubuff, root=0)
except:
    print("Erreur dans la scatter")
    exit()

val_ycbcr = recubuff.reshape((ligne_div,width,channels))
#=================================

# Les composantes Cb (bleu) et Cr (Rouge) sont normalisees :
Cb = (1./255.)*np.array(val_ycbcr[:,:,CB].flat, dtype=np.double)
Cr = (1./255.)*np.array(val_ycbcr[:,:,CR].flat, dtype=np.double)

deb=time.time()
b_Cb = -A.dot(Cb)
b_Cr = -A.dot(Cr)
end = time.time() - deb


im_hsv = im.convert("HSV")
val_hsv = np.array(im_hsv)
deb = time.time()
fix_coul_indices = colorize.search_fixed_colored_pixels(val_hsv)
end = time.time() - deb


# Application de la condition de Dirichlet sur la matrice :    
deb = time.time()
colorize.apply_dirichlet(A, fix_coul_indices)
end = time.time() - deb


deb=time.time()
x0 = np.zeros(Cb.shape,dtype=np.double)
new_Cb = Cb + colorize.minimize(A, b_Cb, x0, niters,epsilon)



deb=time.time()
x0 = np.zeros(Cr.shape,dtype=np.double)
new_Cr = Cr + colorize.minimize(A, b_Cr, x0, niters,epsilon)


# On remet les valeurs des trois composantes de l'image couleur YCbCr entre 0 et 255 :
new_Cb *= 255.
new_Cr *= 255.
intensity *= 255.

shape = (means.shape[0]-2,means.shape[1]-2)
new_image_array = np.empty((shape[0],shape[1],3), dtype=np.uint8)
new_image_array[:,:,0] = intensity[2:-2,2:-2].astype('uint8')
new_image_array[:,:,1] = np.reshape(new_Cb, shape).astype('uint8')
new_image_array[:,:,2] = np.reshape(new_Cr, shape).astype('uint8')

# 在rank 0上创建一个接收最终图像的数组
if rank == 0:
    # 假设我们知道最终图像的形状
    final_image_shape = (height, width, channels)  # 需要根据实际情况调整
    recvbuf = np.empty(final_image_shape, dtype=np.float64)
else:
    recvbuf = None


# 使用gather收集数据
final_image = comm.gather(new_image_array,root=0)


if rank == 0:
    final_image = np.array(final_image)
    new_image = np.vstack(final_image)
    print("final_image shape : ",new_image.shape)
    new_im = Image.fromarray(new_image, mode='YCbCr')
    new_im.convert('RGB').save(output, 'PNG')


















    







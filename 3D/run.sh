#!/bin/bash

# permission as: chmod +x run.sh

# run as:
# ./run.sh 100 100 1000 10 sphere_random_3d_points omp 0.1 1 output/sphere
# ./run.sh 100 100 1000 10 normal_random_3d_points omp " " 1 output/normal

# Configuración inicial
STARTN=$1  # Tamaño inicial
DN=$2      # Incremento del tamaño
ENDN=$3    # Tamaño final
SAMPLES=$4 # Número de muestras
BINARY=$5  # Nombre del binario a ejecutar
ALG=$6   # Parámetro de forma
PROB=$7    # Probabilidad
SEED=${8} # Semilla
OUTFILE=$9 # Archivo de salida

echo "Iniciando ejecución del script para distintos tamaños de entrada y algoritmos"

for ((N=STARTN; N<=ENDN; N+=DN));
do
    echo -n "${N}  " >> "${OUTFILE}"
    for ((k=1; k<=SAMPLES; k++));
    do
        #la seed que se usa es seed*example
        SEED2=$((SEED * k * SAMPLES))
        OUTFILE2="${OUTFILE}_${N}_${SEED2}"
        #echo -n "${OUTFILE2}  "
        echo  "./${BINARY} ${ALG} ${N} $PROB $SEED2 $OUTFILE2"
        value=`./${BINARY} ${ALG} ${N} $PROB $SEED2 $OUTFILE2`
        echo " "
    done
    echo " "
    echo " " >> "${OUTFILE}"
done
echo " " >> "${OUTFILE}"
echo " "

echo "Finalizando ejecución del script"

# Finalizar el trabajo
exit 0

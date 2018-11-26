#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>

// -------- ||||||STRUCTURES||||||| -------

struct matrice{
	long nbLignes;
	long nbColonnes;
	long* data;
};

struct vecteur{
	long dimension;
	long* data;
};

// -------- |||||| DECLARATIONS ||||||| -------

struct matrice* alloMatrice(long nbLignes, long nbColonnes);
struct vecteur* alloVecteur(long dimension);
struct vecteur** alloTabVect(long tailleTab, long dimension);

void setVal(struct matrice* m, long ligne, long colonne, long val);

struct vecteur* getColonne(struct matrice* m, long colonne);
struct vecteur* getLigne(struct matrice* m, long ligne);

long prodVect (struct vecteur* A, struct vecteur* B);
struct matrice* prodMatParallel(struct matrice* A, struct matrice* B);
void prodMatDistrib(int rank, int numprocs, const char* Af, const char* Bf);

struct matrice* lecture(const char* fichier);
void printMatrice(struct matrice* m);

void destructMatrice(struct matrice* m);
void destructVecteur(struct vecteur* vect);
void destructTabVecteur(struct vecteur** vect, long dimension);


// -------- |||||| DESTRUCTS ||||||| -------

void destructMatrice(struct matrice* m){
	free(m->data);
	free(m);
}

void destructVecteur(struct vecteur* vect){
	free(vect->data);
	free(vect);
}

void destructTabVecteur(struct vecteur** vect, long taille){
	for(long i = 0; i < taille; i++) {
		//free(vect[i]->data);
		destructVecteur(vect[i]);
	}
	free(vect);
}

// -------- |||||| ALLOCATIONS ||||||| -------


struct matrice* alloMatrice(long nbLignes, long nbColonnes) {
	struct matrice* m = malloc(sizeof(struct matrice));
	m->data = malloc(sizeof(long) * nbColonnes * nbLignes);
	m->nbLignes = nbLignes;
	m->nbColonnes = nbColonnes;
	return m;
}

struct vecteur* alloVecteur(long dimension) {
	struct vecteur* vect = malloc(sizeof(struct vecteur));
	vect->data = malloc(sizeof(long) * dimension);
	vect->dimension = dimension;
	return vect;
}

struct vecteur** alloTabVect(long tailleTab, long dimension) {
	struct vecteur** tab = malloc(tailleTab * sizeof(struct vecteur*));

	for (long i = 0; i < tailleTab; i++) {
		tab[i] = alloVecteur(dimension);
	}

	return tab;
}

// -------- |||||| PRINT ||||||| -------

void printMatrice(struct matrice* m){
    for(long i = 0; i < pow(m->nbLignes, 2); ++i) {
        if(i > 0 && i % m->nbLignes == 0) {
            printf("\n");
        }
        printf("%ld ", m->data[i]);
    }
	printf("\n");
}

// -------- |||||| LINEARISATION ||||||| -------

void setVal(struct matrice* m, long ligne, long colonne, long val){
    m->data[ligne * m->nbColonnes + colonne] = val;
}

long getVal(struct matrice* m, long ligne, long colonne){
    return m->data[ligne * m->nbColonnes + colonne];
}

// -------- |||||| GET VECTEURS COLONNES ET LIGNES ||||||| -------

struct vecteur* getColonne(struct matrice* m, long colonne){
    struct vecteur* vect = alloVecteur(m->nbLignes);
    for(long i = 0; i < vect->dimension; ++i)
        vect->data[i] = m->data[i * m->nbLignes + colonne];
    return vect;
}

struct vecteur* getLigne(struct matrice* m, long ligne){
    struct vecteur* vect = alloVecteur(m->nbColonnes);
    for(long i = 0; i < vect->dimension; ++i)
        vect->data[i] = m->data[ligne * m->nbLignes + i];
    return vect;
}

// -------- |||||| LECTURE  ||||||| -------

 struct matrice* lecture(const char* fichier){
    FILE* f = fopen(fichier, "r");

    long nbEntiers = 0;
    long d;

    while (fscanf(f, "%ld", &d) != EOF)
        nbEntiers++;

    long nbColonnes = sqrt(nbEntiers); // Comme matrice carrée

    struct matrice* m = alloMatrice(nbColonnes, nbColonnes);
    rewind(f);

    long ligne = 0;
    long colonne = 0;

    while (fscanf(f, "%ld", &d) != EOF) {
    	setVal(m,ligne,colonne,d);
    	colonne++;
    	if (colonne == nbColonnes) {
    		colonne  = 0;
    		ligne++;
    	}
    }

    fclose(f);

    return m;
} 

// -------- |||||| PROD VECTEUR / MATRICE  ||||||| -------

long prodVect (struct vecteur* A, struct vecteur* B){
    long prod = 0;
    #pragma omp parallel for
    for(long i = 0; i < A->dimension; ++i){
        prod += A->data[i] * B->data[i];
    }

    return prod;
}

//Inutilisé dans la version finale
struct matrice* prodMatParallel(struct matrice* A, struct matrice* B) {

	long dimension = A->nbLignes;
	struct matrice* C = alloMatrice(dimension, dimension);

	#pragma omp parallel for
	for(long i = 0; i < dimension; ++i){
		#pragma omp parallel for
		for(long j = 0; j < dimension; ++j){
			#pragma omp parallel for
			for(long k = 0; k < dimension; ++k){
				setVal(C, i, j, getVal(C, i, j) + (getVal(A, i, k) * getVal(B, k, j)));
			}
		}
	}

	return C;
}

void prodMatDistrib(int rank, int numprocs, const char* Af, const char* Bf){

	long dimension;
	long nbDataProc;

	struct matrice* A;
	struct matrice* B;  
	struct vecteur** ligne_a; 
	struct vecteur** colonne_b;
	struct vecteur** res;

	// Version inutilisé => trop lente
/*	if (numprocs == 1) { // Si un seul process => Pas de permutation	
		A = lecture(Af);
		B = lecture(Bf);
        struct matrice* C = prodMatParallel(A,B);
		printMatrice(C);
		return;
	}  

	if (numprocs != 1) { // Si plusieurs process */

	if(rank == 0){ // P0 => Recupère les matrices de base => Les autres process n'y ont pas accès 
		A = lecture(Af);
		B = lecture(Bf);
		dimension = A->nbLignes;
	}

	// Donne la valeur de dimension aux autres process pour les allocations
	MPI_Bcast(&dimension, 1, MPI_LONG, 0, MPI_COMM_WORLD);

	if(rank == 0){ // P0
		// Si N non multiple de P
		if(dimension % numprocs != 0) {
			return;
		}

		// Nombre de lignes et colonnes gérées par un process
		nbDataProc = dimension / numprocs;
		
		// Matrice resultat
		struct matrice* C = alloMatrice(dimension, dimension);

		//Allocation des vecteurs 
		ligne_a = alloTabVect(nbDataProc, dimension);
		colonne_b = alloTabVect(nbDataProc, dimension);
		res = alloTabVect(nbDataProc, dimension);

		// Première(s) ligne(s)/colonne(s) 
		#pragma omp parallel for
		for(long i = 0; i < nbDataProc; ++i){
			ligne_a[i] = getLigne(A, i);
			colonne_b[i] = getColonne(B, i);
		}
		// Distribution des données aux autres processeurs => Initial
		for(long proc = 1; proc < numprocs; ++proc){
			for(long numLC = 0; numLC < nbDataProc; ++numLC){ // numLC => Num ligne/colonne
					
				struct vecteur* ligneProcess = getLigne(A, numLC + proc * nbDataProc);
				struct vecteur* colonneProcess = getColonne(B, numLC + proc * nbDataProc);

				//Envoie à chaque processeur sa/ses ligne(s)/colonne(s) de matrice correspondante
				MPI_Send(ligneProcess->data, dimension, MPI_LONG, proc, 0, MPI_COMM_WORLD);
				MPI_Send(colonneProcess->data, dimension, MPI_LONG, proc, 0, MPI_COMM_WORLD);

				destructVecteur(ligneProcess);
				destructVecteur(colonneProcess);
			}
		}

		// Calcul de la/des première(s) colonne(s) matrice
		#pragma omp parallel for
		for(long colonne = 0; colonne < nbDataProc; ++colonne){
			for(long ligne = 0; ligne < nbDataProc; ++ligne){
					res[colonne]->data[ligne] = prodVect(ligne_a[ligne], colonne_b[colonne]);
				}
			}
			
		// Permutation des lignes
		for(long i = 0; i < numprocs - 1; ++i){
			for(long numLC = 0; numLC < nbDataProc; ++numLC){
				//Chaque processeur envoie sa/ses ligne(s) au suivant et reçoit une/des nouvelle(s) ligne(s) du précédent
				MPI_Send(ligne_a[numLC]->data, dimension, MPI_LONG, 1, 0, MPI_COMM_WORLD);
				MPI_Recv(ligne_a[numLC]->data, dimension, MPI_LONG, numprocs - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				// Calcul local du resultat après permutation
				#pragma omp parallel for
				for(long j = 0; j < nbDataProc; ++j){
					long r = prodVect(ligne_a[numLC], colonne_b[j]);
					res[j]->data[(i + 1) * nbDataProc + numLC] = r;
				}
			}
		}

		// Remplissage de la matrice C avec les résultats obtenus 
		if(nbDataProc == 1){ // Si on a autant de process que de lignes/colonnes
			#pragma omp parallel for
			for(long i = 0; i < dimension; ++i)
				setVal(C, (dimension - i) % dimension, 0, res[0]->data[i]); 
			
		} else { //Sinon 
			
		#pragma omp parallel for
        for(long colonne = 0; colonne < nbDataProc; ++colonne){
            for(long ligne = 0; ligne < dimension; ligne += nbDataProc){
                long ind = 0;
                    if(ligne < nbDataProc){
                    	ind = ligne;
            		} else if(ligne >= nbDataProc && ligne <= dimension - nbDataProc){
							ind = dimension - ligne;
					} else if(ligne == dimension - nbDataProc){
							ind = nbDataProc + colonne;
					}
					//Remplissage Matrice résultat
					for(long numLC = 0; numLC < nbDataProc; ++numLC){
						setVal(C, ligne + numLC, colonne, res[colonne]->data[ind + numLC]);
					}
				}
			}
		}

		// Rassemble données autres process
		for(long proc = 1; proc < numprocs; ++proc){
			for(long numLC = 0; numLC < nbDataProc; ++numLC){
				MPI_Recv(res[numLC]->data, dimension, MPI_LONG, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}

			//Remplissage de la matrice C avec les résultats obtenus 
			if(nbDataProc == 1){ // Si autant de process que de lignes/colonnes => n = p 
				for(long i = 0; i < dimension; ++i){
					setVal(C, (dimension - i + proc) % dimension, proc, res[0]->data[i]);
				}

			} else { // Si n multiple de p mais n != p
				#pragma omp parallel for
				for(long colonne = 0; colonne < nbDataProc; ++colonne){
					for(long ligne = 0; ligne < dimension; ligne += nbDataProc){
						long ind = ligne;
						if(ligne >= nbDataProc && ligne <= dimension - nbDataProc){
							ind = dimension - ligne;
						} else if(ligne == dimension - nbDataProc){
							ind = nbDataProc + colonne;
						}

						long ind_ligne = (dimension + ligne + (proc * nbDataProc)) % dimension;
						long ind_colonne = colonne + proc * nbDataProc;

						for(long numLC = 0; numLC < nbDataProc; ++numLC){
							setVal(C, ind_ligne + numLC, ind_colonne, res[colonne]->data[ind + numLC]);
						}
					}
				}
			}
		}

		printMatrice(C);

		destructMatrice(A);
		destructMatrice(B);
		destructMatrice(C);
	}

	else { // P != P0

		// Nombre de lignes et colonnes gérées par un process
		nbDataProc = dimension / numprocs;

		//Allocation des vecteurs 
		ligne_a = alloTabVect(nbDataProc, dimension);
		colonne_b = alloTabVect(nbDataProc, dimension);
		res = alloTabVect(nbDataProc, dimension);

		// Reçois première(s) ligne(s) et colonne(s)
		for(long i = 0; i < nbDataProc; ++i){
			MPI_Recv(ligne_a[i]->data, dimension, MPI_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(colonne_b[i]->data, dimension, MPI_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// Calcule premier res du processeur
		#pragma omp parallel for
		for(long colonne = 0; colonne < nbDataProc; ++colonne){
			for(long ligne = 0; ligne < nbDataProc; ++ligne){
				res[colonne]->data[ligne] = prodVect(ligne_a[ligne], colonne_b[colonne]);
			}		
		}

		long cpt = 0;

		// Reçois d'autres lignes et calcule données reçues et sauvegarde avant de les renvoyer à P0
		for(long i = 0; i < numprocs - 1; ++i){
			cpt = 0;

			for(long numLC = 0; numLC < nbDataProc; ++numLC){
				struct vecteur* lignea = alloVecteur(dimension);
				MPI_Recv(lignea->data, dimension, MPI_LONG, (numprocs + rank - 1) % numprocs, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(ligne_a[cpt]->data, dimension, MPI_LONG, (numprocs + rank + 1) % numprocs, 0, MPI_COMM_WORLD);

				ligne_a[cpt] = lignea;
				++cpt;
					
				#pragma omp parallel for
				for(long j = 0; j < nbDataProc; ++j){
					long r = prodVect(ligne_a[numLC], colonne_b[j]);
					res[j]->data[numLC + (i+ 1) * nbDataProc] = r;
				}
			}
		}

		//Renvoie res final
		for(long numLC = 0; numLC < nbDataProc; ++numLC){
			MPI_Send(res[numLC]->data, dimension, MPI_LONG, 0, 0, MPI_COMM_WORLD);
		}
	}

	//	destructTabVecteur(ligne_a, nbDataProc);
	//	destructTabVecteur(colonne_b, nbDataProc);
		destructTabVecteur(res, nbDataProc);
	//}
}

// -------- |||||| MAIN  ||||||| -------


int main(int argc, char** argv) {
	int numprocs;
	int rank;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    prodMatDistrib(rank, numprocs, argv[1], argv[2]);

	MPI_Finalize();

    return 0;
}
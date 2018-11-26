#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <limits.h>
#include <math.h>


// -------- ||||||STRUCTURES||||||| -------

struct tablo {
    long* tab;
    long size;
};

struct maxData {
    struct tablo* maxSubSeq;
    long max;
};

struct mBornes{
    long dep;
    long fin;
};

// -------- ||||||DECLARATIONS||||||| -------

void destruct(struct tablo* tab);
long sum(long a, long b);
long max(long a, long b);
struct tablo* alloTablo(long size);
void printRes(struct tablo* tmp);
struct tablo* reverse(long ind, long size, struct tablo* src);
void montee(struct tablo* source, struct tablo* destination, long (*fun)(long,long), long init);
void descente(struct tablo* a, struct tablo* b, long (*fun)(long, long), long init);
void final(struct tablo * a, struct tablo *b, long (*fun)(long,long));
struct tablo* prefixSum(struct tablo* source);
struct tablo* prefixMax(struct tablo* source);
struct tablo* suffixSum(struct tablo* src);
struct tablo* suffixMax(struct tablo* src);
struct tablo* tableauM(struct tablo* src, struct tablo* prefixSum, struct tablo* prefixMax, struct tablo* suffixSum, struct tablo* suffixMax);
struct maxData* alloMaxData(long size);
void fillWith(struct maxData* sm, struct tablo* with, long dep, long fin);
struct mBornes* getmBornes(struct tablo* src, struct tablo* M, long valMax);
struct maxData* sousSeqMax(struct tablo* source, struct tablo* M);
struct tablo* genRes(struct tablo* src, struct tablo* M);
struct tablo* lecture(const char* fichier);


// -------- ||||||DESTRUCT||||||| -------

void destruct(struct tablo* tabl){
    free(tabl->tab);
    free(tabl);
}
// -------- |||||||FONCTION SUM ET MAX|||||| -------

long sum(long a, long b) {
    return a + b;
}

//---- 

long max(long a, long b) {
    if (a > b) return a;
    return b;
}

// -------- ||||||ALLO ET PRINT RESULTAT||||||| -------


struct tablo* alloTablo(long size) {
    struct tablo* tmp = malloc(sizeof(struct tablo));
    tmp->size = size;
    tmp->tab = malloc(size * sizeof(long));
    return tmp;
}

//----

void printRes(struct tablo* tmp) {
    for (long i = 0; i < tmp->size - 1; ++i) {
        printf("%ld ", tmp->tab[i]);
    }
    
    printf("%ld\n", tmp->tab[tmp->size - 1]);
}

// -------- |||||||REVERSE|||||| -------

struct tablo* reverse(long ind, long size, struct tablo* src){
    struct tablo* rev = alloTablo(size);

    #pragma omp parallel for
    for(long i = src->size - 1; i >= ind; --i){
        rev->tab[src->size - 1 - i] = src->tab[i];
    }

    return rev;
}

// -------- ||||||PARALLEL PREFIX||||||| -------

void montee(struct tablo* source, struct tablo* destination, long (*fun)(long,long), long init) {
    long n = source->size;
    long m = log2(n);
    
    #pragma omp parallel for
    for(long i = 0; i < source->size; i++){ //Rempli avec minimum
        destination->tab[i] = init; //Minimum
    }
    
    //Rempli destination avec val de source
    #pragma omp parallel for
    for(long i = 0; i < source->size; i++){
        destination->tab[i + source->size] = source->tab[i];
    }
    
    // Boucle de calcul pour la montée dans l'arbre/tableau
    for (long k = m-1 ; k >= 0; k--) {
        #pragma omp parallel for
        for(long l = (long) pow(2,k); l <= (long) (pow(2,(k+1))-1); l++) {
            destination->tab[l] = fun(destination->tab[2*l], destination->tab[2*l + 1]);
        }
    }
}

//---- 

void descente(struct tablo* a, struct tablo* b, long (*fun)(long,long), long init) {
    long n = a->size;
    long m = log2(n);

    #pragma omp parallel for
    for(long i = 0; i < b->size; i++){ //Rempli avec minimum
        b->tab[i] = init; //Minimum
    }
    
    for (long k = 1; k < m; k++) {
    #pragma omp parallel for
        for (long i = (long) pow(2, k); i < (long) pow(2, k+1); i++) {
            if (i % 2 == 0) {
                b->tab[i] = b->tab[i / 2];
            }
            else {
                b->tab[i] = fun(b->tab[i / 2], a->tab[i - 1]);
            }
        }
    }
}

//----

void final(struct tablo* a, struct tablo* b, long (*fun)(long,long)) {
    long n = a->size;
    long m = log2(n);
    
    #pragma omp parallel for
    for (long i = (long) pow(2, m - 1); i < (long) pow(2, m); i++) {
        b->tab[i] = fun(b->tab[i],a->tab[i]);
    }
}

// -------- ||||||PREFIX||||||| -------


struct tablo* prefixSum(struct tablo* source){
    struct tablo * a = alloTablo(source->size*2);
    struct tablo * b = alloTablo(source->size*2);

    montee(source, a, sum, 0);
    descente(a, b, sum, 0);
    final(a,b,sum);
    destruct(a);
    return b;
}

//---- 

struct tablo* prefixMax(struct tablo* source){
    struct tablo * a = alloTablo(source->size*2);
    struct tablo * b = alloTablo(source->size*2);

    montee(source, a, max, LONG_MIN);
    descente(a, b, max, LONG_MIN);
    final(a,b,max);
    destruct(a);
    return b;
}

// -------- |||||||SUFFIX|||||| -------


struct tablo* suffixSum(struct tablo* src){
    struct tablo* reverseTab = reverse(0, src->size, src);
    struct tablo* reversePrefix = prefixSum(reverseTab);
    struct tablo* suffix = reverse(src->size, src->size, reversePrefix);
    
    destruct(reverseTab);
    destruct(reversePrefix);
    return suffix;
}

//---- 

struct tablo* suffixMax(struct tablo* src){
    struct tablo* reverseTab = reverse(0, src->size, src);
    struct tablo* reversePrefix = prefixMax(reverseTab);
    struct tablo* suffix = reverse(src->size, src->size, reversePrefix);
    
    destruct(reverseTab);
    destruct(reversePrefix);
    return suffix;
}

// -------- ||||||TABLEAU M||||||| -------


struct tablo* tableauM(struct tablo* src, struct tablo* prefixSum, struct tablo* prefixMax, struct tablo* suffixSum, struct tablo* suffixMax){
    struct tablo* M = alloTablo(src->size);
    
    #pragma omp parallel for
    for(long i = 0; i < src->size; i++){
        long ind = i + src->size; // pour pointer sur la bonne position
        long ms = prefixMax->tab[ind] - suffixSum->tab[i] + src->tab[i];
        long mp = suffixMax->tab[ind] - prefixSum->tab[ind] + src->tab[i];
        M->tab[i] = ms + mp - src->tab[i];
    }
    
    return M;
}

// -------- |||||||PARALLEL FILL WITH|||||| -------

void fillWith(struct maxData* sm, struct tablo* with, long dep, long fin){
    #pragma omp parallel for
    for(long i = dep; i <= fin; i++){
        sm->maxSubSeq->tab[i - dep] = with->tab[i];
    }
}

// -------- |||||||SOUS-SEQ MAX|||||| -------

struct maxData* alloMaxData(long size){
    struct maxData* maxp = malloc(sizeof(struct maxData));
    struct tablo* maxSubSeq = alloTablo(size);
    maxp->maxSubSeq = maxSubSeq;
    maxp->max = LONG_MIN;
    return maxp;
}

//---- 

struct mBornes* getmBornes(struct tablo* src, struct tablo* M, long valMax){
    long dep = -1;
    long fin = -1;

	#pragma omp parallel for
    for(long i = 1; i < M->size - 1; i++){
        if(dep == -1){
            if(M->tab[i - 1] == valMax && i - 1 == 0){
                dep = i - 1;
            }

            if(M->tab[i] == valMax && M->tab[i - 1] != valMax){
                dep = i;
            }
        }

        if(fin == -1){
            if(M->tab[i + 1] == valMax && i + 1 == M->size - 1){
                fin = i + 1;
            }

            if(M->tab[i] == valMax && M->tab[i + 1] != valMax){
                fin = i;
            }
        }   
    }

    if(dep < 0 || fin < 0){
        if(M->tab[0] == valMax){
            dep = 0;
            fin = 0;
        }
        else if(M->tab[M->size - 1] == valMax){
            dep = M->size - 1;
            fin = dep;
        } 
    }

    struct mBornes* mBornes = malloc(sizeof(struct mBornes));
    mBornes->dep = dep;
    mBornes->fin = fin;
    return mBornes;
}

//---- 

struct maxData* sousSeqMax(struct tablo* source, struct tablo* M){
    // obtenir la valeur maximale de M
    struct tablo* suffixM = suffixMax(M);
    long valMax = suffixM->tab[0];
    destruct(suffixM);

    // obtenir la sous sequence
    struct mBornes* mBornes = getmBornes(source, M, valMax);
    long maxSubSeqSize = mBornes->fin - mBornes->dep + 1;

    struct maxData* SM = alloMaxData(maxSubSeqSize);
    fillWith(SM, source, mBornes->dep, mBornes->fin);
    SM->max = valMax;

    free(mBornes);
    return SM;
}

// -------- |||||||CREATION RESULTAT|||||| -------

struct tablo* genRes(struct tablo* src, struct tablo* M) {
    struct maxData* SM = sousSeqMax(src, M);

    long size = SM->maxSubSeq->size + 1;
    struct tablo* res = alloTablo(size);
    res->tab[0] = SM->max;

    #pragma omp parallel for
    for(long i = 1; i < size; i++) {
        res->tab[i] = SM->maxSubSeq->tab[i - 1];
    }

    destruct(SM->maxSubSeq);
    free(SM);
    return res;
}

// -------- |||||||LECTURE FICHIER|||||| -------

 struct tablo* lecture(const char* fichier){
    FILE* f = fopen(fichier, "r");

    long nbEntiers = 0;
    long d;
    long i = 0;

    // Compter le nombre d'entiers pour allouer bonne taille
    while (fscanf(f, "%ld", &d) != EOF)
        nbEntiers++;

    rewind(f);
    struct tablo* tab = alloTablo(nbEntiers);

    while (fscanf(f, "%ld", &tab->tab[i++]) != EOF) ;
    return tab;
} 

// -------- ||||||||||||| -------

int main(int argc, char** argv){
    if(argc != 2){
        return -1;
    }

    struct tablo* src = lecture(argv[1]);
    struct tablo* preSum = prefixSum(src);
    struct tablo* suSum = suffixSum(src);
    struct tablo* suMax = suffixMax(preSum);
    struct tablo* preMax = prefixMax(suSum);
    struct tablo* M = tableauM(src, preSum, preMax, suSum, suMax);
    struct tablo* res = genRes(src, M);

    printRes(res);

    destruct(src);
    destruct(preSum);
    destruct(suSum);
    destruct(suMax);
    destruct(preMax);
    destruct(M);
    destruct(res);

    return 0;
}

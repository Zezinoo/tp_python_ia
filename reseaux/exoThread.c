// A compiler comme ceci : gcc prog.c -lpthread -o prog
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>

int counter;


void *runDuThread (void *p) {  // fonction principale du thread 1
	int i ;
	for ( i=0 ; i<1000000; i++ ){
		//printf ("je suis le thread n° %s\n", (char *)p );
		counter++;
		//usleep(100000) ; // attente de 1e6 us
	}
	printf("thread1 , counter = %d\n" , counter);
	pthread_exit(0);
}


void *runDuThread20 (void *p) {  // fonction principale du thread 2
	int i ;
	for ( i=0 ; i<1000000 ; i++ ){
		//printf ("je suis le thread n° %s\n", (char *)p );
		counter++;
		//usleep(100000) ; // attente de 1e6 us
	}
	printf("thread2 , counter = %d\n" , counter);
	pthread_exit(0);
}

int main (int argc, char **argv) {
	pthread_t th1, th2 ; int ret ;
	ret = pthread_create (&th1, NULL, runDuThread, (void *)"1") ; // création thread 1
	ret = pthread_create (&th2, NULL, runDuThread20, (void *)"2") ; // création thread 2

	if (ret != 0) {
		perror ("Pb creation du thread\n") ; exit(0) ;
	}
	pthread_join(th1, (void **)&ret); // attente thread 1
	pthread_join(th2, (void **)&ret); // attente thread 2
	printf("main , counter = %d\n" , counter);
	printf ("Fin main") ;
	return 0 ;
}

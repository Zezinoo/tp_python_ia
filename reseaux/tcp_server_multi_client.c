/**
 * Usage: ./tcp_server <port_no>
 * 
 * TCP server that waits for a client to connects.
 * Reads messsage sent by client, turns it into upper case and sends it back.
 * Disconnects when client sends "bye".
 * 
 * @author  Philippe Lefebvre, ENSICAEN
 * @author  Dimitri Boudier,   ENSICAEN
 */

#include <errno.h>                      // Error management
#include <sys/socket.h>                 // Socket
#include <arpa/inet.h>                  // Internet
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>                     // Removes warning of implicit declaration when closing socket
#include <ctype.h>
#include <pthread.h>

#define MAX_LENGTH 64                   // Maximum length for sent and received messages

void * runDuThread(void* p);
void * envoyerClavier(void *p);

int n_clients;
int socket_array[100];


int main(int argc, char **argv) {
    
    // Check the number of arguments : "tcp_server port_no"
    int server_port_no;
    if (argc != 2) {
        printf ("ARGUMENT ERROR. USAGE: ./tcp_server <port_no>\n");
        exit (-1);
    } else {
        sscanf(argv[1], "%d", &server_port_no);     // argv[1] contains the server's port number
    }

    
    // Open socket in connected mode (TCP). 6 is TCP number ( cf. file /etc/protocols )
    int network_socket = socket(AF_INET, SOCK_STREAM, 6); // We are using IP and TCP
    if ( network_socket < 0 ) {
        perror("\nERROR WHILE OPENING LISTENING SOCKET:\n");
        exit(-1);
    }


    // Set server parameters
    struct sockaddr_in server_address, client_address;          // Structure that will contain the server and client address
    socklen_t address_length = sizeof(struct sockaddr_in);      // Length of sockaddr_in structure

    server_address.sin_family       = AF_INET;                  // Server address is Internet-type
    server_address.sin_port         = htons(server_port_no);    // Server port number
    server_address.sin_addr.s_addr  = INADDR_ANY;               // Server can accept any incoming IP address

    // Syscalls like socket are thin wrappers around a kernel system call
    // Bind socket to listening port
    int socket_error = bind(network_socket, (struct sockaddr *)&server_address, sizeof(struct sockaddr_in));
    if (socket_error < 0) {
        perror ("ERROR WHILE BINDING SOCKET:");
        exit(-1);
    }

    // Operating System can accept connections
    // Waiting Connection buffer size is set to 1 
    socket_error = listen(network_socket, 1);
    if (socket_error < 0) {
        perror("\nERROR WHILE SETTING UP LISTENING SOCKET:\n");
        exit(-1);
    }
    
    // Wait for a connection (the program is paused if no pending connection)
    // then creates a dialogue socket when a client connects

    pthread_t thClavier; int retClavier;
    retClavier = pthread_create(&thClavier , NULL , envoyerClavier , (void*)socket_array);
        
    

    while(1){
        int client_socket = accept(network_socket, (struct sockaddr *) &client_address, &address_length); 
        socket_array[n_clients] = client_socket;
        printf("Client connected from %s\n" , inet_ntoa(client_address.sin_addr));
        pthread_t th1 ; int ret ;
        ret = pthread_create (&th1, NULL, runDuThread, (void *)&client_socket) ; // création thread 1
        n_clients++;
    } //While loop always runs and accepts new connections
    
}


void * envoyerClavier(void *p){
    //printf("Im sending somet);
    while (1)
    {
        char sent_msg[MAX_LENGTH];
        fgets(sent_msg, MAX_LENGTH,stdin);
        int msg_len = strlen(sent_msg);
        //int socket_array[100] = (int)p;
        for(int i = 0 ; i < n_clients;i++){
            send(socket_array[i] , sent_msg , msg_len ,0 );
        }
    }
    
}


void *runDuThread (void*p){

    char message[MAX_LENGTH]; int msg_length;
    int client_socket = *(int *)p; // casts p into integer then turns it into integer pointer
    do {
        // Read string from client through the dialogue socket
        msg_length = recv( client_socket, message, MAX_LENGTH-1, 0 );
        if (msg_length == 0) {
            printf("CLIENT HAS DISCONNECTED\n\n");
            exit(0);
        }
        if (msg_length < 0) {
            perror("ERROR ON DATA RECEPTION:\n");
            exit(-1);
        }
        
        message[msg_length] = '\0';                                 // Puts '\0' as last character to turn this array into a string
        
        int i;
        for(i = 0 ; i < msg_length ; i++){
            message[i] = toupper(message[i]);
        }
        
        printf( "Received %d chars: %s", msg_length, message );     // Display received string onto console
        send(client_socket, message, msg_length, 0);                // Send updated message back to the client
        
    } while ( strncmp(message, "BYE", 3) );      // Loop until the sent message is "BYE"
    
    printf ("Client has disconnected\n");
    close(client_socket);
}

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

#define MAX_NUMBER_CLIENTS 100
#define MAX_LENGTH 64                   // Maximum length for sent and received messages

int n_clients = 0;
int socket_array[MAX_NUMBER_CLIENTS];

typedef struct {
    int client_socket;
    int client_id;
} ThreadArgs; //defining struct to pass multiple parameters to the thread function


void broadcastToAll(const char *msg , int exception) {
    // Helper function to send message to all valid socket array components
    int msg_len = strlen(msg);
    for (int i = 0; i < n_clients; i++) {
        if (i != exception && socket_array[i] > 0) {
            send(socket_array[i], msg, msg_len, 0);
        }
    }
}

void * envoyerClavier(void *p){
    while (1)
    {   printf("> ");
        fflush(stdout); //empties stdout buffer to ensure prompt is displayed
        char sent_msg[MAX_LENGTH];
        fgets(sent_msg, MAX_LENGTH,stdin);

        char full_msg[MAX_LENGTH + 32];
        snprintf(full_msg, sizeof(full_msg), "[HOST MESSAGE] : %s", sent_msg); //concatenating strings for logging

        broadcastToAll(full_msg , -99);
    }
    
}


void *runDuThread (void*p){

    char message[MAX_LENGTH]; int msg_length;
    // Unpacking ThreadArgs struct to get the client id for logging and socket for TCP communication
    ThreadArgs *args = (ThreadArgs*)p;
    int client_socket = args->client_socket; 
    int client_id = args->client_id;

    // Welcoming new user routine
    char username[MAX_LENGTH];
    char welcome_msg[MAX_LENGTH] = "Welcome to the server! Please input your username:";
    send(client_socket , welcome_msg , strlen(welcome_msg),0);
    int username_len = recv(client_socket,username,MAX_LENGTH -1 ,0);
    username[strcspn(username, "\n")] = '\0'; // Prevents newline on connection message by replacing trailing \n with \0

    char connection_msg[MAX_LENGTH + 32];
    snprintf(connection_msg, sizeof(connection_msg),
         "Client %.60s has connected.", username);

    broadcastToAll(connection_msg , -99); // Broadcast to all, including user

    do {
        // Read string from client through the dialogue socket
        msg_length = recv( client_socket, message, MAX_LENGTH-1, 0 );
        if (msg_length == 0) {
            printf("Client %d disconnected abruptly\n", client_id);
            fflush(stdout);
            socket_array[client_id - 1] = -1; 
            close(client_socket);
            pthread_exit(NULL);
                }
        if (msg_length < 0) {
            perror("ERROR ON DATA RECEPTION:\n");
            exit(-1);
        }
        
        message[msg_length] = '\0';                                 // Puts '\0' as last character to turn this array into a string
        
        
        printf( "Received %d chars from %s: %s", msg_length,username, message );    // Display received string onto console
        fflush(stdout);
        printf( "\n> ");
        fflush(stdout);

        // Broadcast message with username to all users , except current user. Note that the socket array index is client_id -1
        char full_msg[sizeof(message) + sizeof(username) + 6];
        snprintf(full_msg , sizeof(full_msg), "[%s]: %s",username,message);
        broadcastToAll(full_msg , client_id - 1);

    } while ( strncmp(message, "bye", 3) );      // Loop until the sent message is "bye"
    
    // Disconnection Routine. Displays message to host, close socket on host end , exit thread
    printf("Client %s disconnected", username);
    fflush(stdout);
    socket_array[client_id - 1] = 0; // putting current socket array position as invalid to broadcast
    close(client_socket); // closing socket on client side
    char disconnection_msg[MAX_LENGTH + 64];
    snprintf(disconnection_msg, sizeof(disconnection_msg),"Client %s has disconnected.\n", username); 
    broadcastToAll(disconnection_msg, client_id - 1);
    pthread_exit(NULL);
}




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
        printf("> ");
        fflush(stdout);
        n_clients ++;
        printf("Created with number %d\n", n_clients);
        
        // Creating ThreadArgs
        pthread_t th1 ; int ret ;
        ThreadArgs *args = malloc(sizeof(ThreadArgs));
        args->client_socket = client_socket;
        args->client_id = n_clients;

        ret = pthread_create (&th1, NULL, runDuThread, args) ;
    } //While loop always runs and accepts new connections
    
}



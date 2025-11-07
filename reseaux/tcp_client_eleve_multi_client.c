/**
 * Usage: ./tcp_client <server_ip_address> <server_port_no>
 * 
 * TCP client that connects to a server, reads text from console, sends to server, displays server response.
 * Disconnects from server when "bye" is sent.
 * 
 * @author  Philippe Lefebvre, ENSICAEN
 * @author  Dimitri Boudier,   ENSICAEN
 */


#include <errno.h>                      // Error management
#include <sys/socket.h>                 // Socket
#include <arpa/inet.h>                  // Internet
#include <stdio.h>                      // I/O
#include <stdlib.h>
#include <string.h>
#include <unistd.h>                     // Removes warning of implicit declaration when closing socket
#include <string.h>
#include <pthread.h>

#define MAX_LENGTH 64                   // Maximum length for sent and received messages


void *recvThread(void *p){
    int client_socket = *(int *)p; //casts p into integer then turn it into integer pointer
    char recv_msg[MAX_LENGTH];

    while (1)
    {
        // Read up to MAX_LENGTH characters from server and put '\0' as last character to turn this array into a string
        int msg_length = recv( client_socket, recv_msg, MAX_LENGTH-1, 0 );
        if (msg_length == 0) {
            printf("\nHOST HAS DISCONNECTED\n\n");
            exit(0);
        }
        if (msg_length < 0) {
            perror("\n HOST ERROR ON DATA RECEPTION:\n");
            exit(-1);
        }
        recv_msg[msg_length] = '\0';
        // Display received message to console
        recv_msg[strcspn(recv_msg, "\n")] = '\0'; // remove extra newline

        printf("\r%s\n[YOU]> ", recv_msg);
        fflush(stdout);
    }
    
}

void main( int argc, char **argv) {

    // Check the number of arguments : "tcp_client xxx.xxx.xxx.xxx yyy"
    char*  server_IP_address;
    int    server_port_no;
    if (argc !=3 ) {
        printf ("ARGUMENT ERROR. USAGE: ./tcp_client IP_address no_port \n");
        exit (-1);
    }
    else {
        server_IP_address = argv[1];
        sscanf(argv[2], "%d", &server_port_no);
    }
  

    // Open socket in connected mode (TCP). 6 is TCP number ( cf. file /etc/protocols )
    // *** CODE ICI EN FONCTION DU COMMENTAIRE PRECEDENT ***    
    int client_socket = socket(AF_INET, SOCK_STREAM, 6); // We are using IP and TCP
    if ( client_socket < 0 ) {
        perror("\nERROR WHILE OPENING LISTENING SOCKET:\n");
        exit(-1);
    }



    

    // Set server parameters
    struct sockaddr_in server_address;
    server_address.sin_family       = AF_INET;                          // Server address is Internet-type
    server_address.sin_port         = htons (server_port_no);           // Set server port number, hton converts unsigned short integer into network byte order 
    server_address.sin_addr.s_addr  = inet_addr(server_IP_address);     // Set server IP address , inet_addr converts string (decimal-and-dots notation) into binary form
    

    // Connect socket to server
    int socket_error = connect( client_socket, (struct sockaddr *) &server_address, sizeof (struct sockaddr_in) );
    if (socket_error < 0) {
        perror ("ERROR WHILE CONNECTING SOCKET:\n");
        exit(1);
    }

    //Routine to continuously listen to server and display host messages

    pthread_t recv_thread; int retServerListen;
    retServerListen = pthread_create(&recv_thread , NULL , recvThread , &client_socket);

    // Main routine: talk to server and listen to its answer
    char sent_msg[MAX_LENGTH];
    char recv_msg[MAX_LENGTH];
    int  recv_length;
    printf("Ready to send text to server!\n\n");
    printf ("> ");
    fflush(stdout);
    do {
        // Read string from console
        fgets(sent_msg, MAX_LENGTH, stdin);
        printf ("[YOU]> ");
        fflush(stdout);
        
        // *** CODER ICI EN FONCTION DES COMMENTAIRES SUIVANTS ***
        
        // Send message to the server through socket

        int msg_length = strlen(sent_msg);
        send(client_socket, sent_msg, msg_length, 0);  // Send the message to server

    } while( strncmp(sent_msg, "bye", 3) );     // Loop until the sent message is "bye"
    
    printf ("Disconnection\n");
    close(client_socket);
}

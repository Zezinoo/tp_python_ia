/**
 * Usage: ./udp_server <port_no>
 * 
 * UDP server that waits for a client to connects.
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
#include <ctype.h>                      // toupper() function

#define MAX_LENGTH 64                   // Maximum length for sent and received messages


void toUpperCase(char *str) {
    for (int i = 0 ; i < strlen(str) ; i++)
        str[i] = toupper( str[i] );
}


int main( int argc, char **argv) {


    // Check the number of arguments : "udp_server port_no"
    int  server_port_no;
    if (argc != 2) {
        printf ("ARGUMENT ERROR. USAGE: ./udp_server port_no\n");
        exit (-1);
    } 
    else {
        sscanf(argv[1], "%d", &server_port_no);        // argv[1] contains the server's port number
    }
    

    // Open socket in datagram mode (UDP). 17 is UDP number ( cf. file /etc/protocols )
    int udp_socket = socket(AF_INET, SOCK_DGRAM, 17);
    if ( udp_socket < 0 ) {
        perror("\nERROR WHILE OPENING SOCKET:\n");
        exit(-1);
    }
  

    // Set server parameters
    struct sockaddr_in server_address, client_address;
    socklen_t address_length = sizeof(struct sockaddr_in);
    server_address.sin_family       = AF_INET;                  // Server address is Internet-type
    server_address.sin_port         = htons(server_port_no);    // Server port number
    server_address.sin_addr.s_addr  = INADDR_ANY;               // Server can have any IP address


    // Bind socket to listening port
    int socket_error = bind(udp_socket, (struct sockaddr *)&server_address, sizeof(struct sockaddr_in));
    if (socket_error < 0) {
        perror ("ERROR WHILE BINDING SOCKET:");
        exit(-1);
    }
  

    int  msg_length;
    char message[MAX_LENGTH];
    printf ("Server started\n\n");
    do {
        // Read string from client through the dialogue socket
        msg_length = recvfrom(udp_socket, message, MAX_LENGTH, 0, (struct sockaddr *)&client_address,  &address_length);
        //
        // In disconnected mode (UDP)
        // there is no need for checking what has been sent
        // but it can be done if you want to
        //
        
        message[msg_length] = '\0';
        printf( "Received %d chars: %s\n", msg_length, message );
        toUpperCase( message );

        sendto(udp_socket, message, msg_length, 0, (struct sockaddr *) &client_address, address_length);
    
    } while ( strncmp(message, "BYE", 3));       // Loop until the sent message is "BYE"
    
    printf ("Client has disconnected\n");
    
    close(udp_socket);
}

/**
 * Usage: ./udp_client <server_ip_address> <server_port_no>
 * 
 * UPD client that connects to a server, reads text from console, sends to server, displays server response.
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


#define MAX_LENGTH 64                   // Maximum length for sent and received messages


void main( int argc, char **argv) {
    
    // Check the number of arguments : "udp_client xxx.xxx.xxx.xxx yyy"
    char*  server_IP_address;
    int    server_port_no;
    if (argc !=3 ) {
        printf ("ARGUMENT ERROR. USAGE: ./udp_client IP_address no_port \n");
        exit (-1);
    }
    else
    {
        server_IP_address = argv[1];            // argv[1] contains the server's IP address in string format
        sscanf(argv[2], "%d", &server_port_no); // argv[2] contains the server's port number
    }
    

    // Set server parameters
    struct sockaddr_in server_address;
    socklen_t address_length = sizeof(struct sockaddr_in);
    server_address.sin_family       = AF_INET;                          // Server address is Internet-type
    server_address.sin_port         = htons (server_port_no);           // Set server port number, hton converts unsigned short integer into network byte order 
    server_address.sin_addr.s_addr  = inet_addr(server_IP_address);     // Set server IP address , inet_addr converts string (decimal-and-dots notation) into binary form
    
    // Open socket in disconnected mode (UDP). 17 is UDP number ( cf. file /etc/protocols )
    // *** CODER ICI EN SUIVANT LE COMMENTAIRE PRECEDENT ***
    int udp_socket = socket(AF_INET, SOCK_DGRAM, 17);
    if ( udp_socket < 0 ) {
        perror("\nERROR WHILE OPENING SOCKET:\n");
        exit(-1);
    }
    
    
    char sent_msg[MAX_LENGTH];
    char recv_msg[MAX_LENGTH];
    int  recv_length;
    printf("Ready to send text to server!\n\n");
    do {
        // *** CODER ICI EN SUIVANT LES COMMENTAIRES ***

        // Read string from console
        printf ("> ");
        fgets(sent_msg, MAX_LENGTH, stdin);
        int msg_length = strlen(sent_msg);
        // Send message to the server through socket

        sendto(udp_socket, sent_msg, msg_length, 0, (struct sockaddr *) &server_address, address_length);

        // Read up to MAX_LENGTH characters from server and put '\0' as last character to turn this array into a string
        recv_length = recvfrom(udp_socket, recv_msg, MAX_LENGTH, 0, (struct sockaddr *)&server_address,  &address_length);
        printf( "Received %d chars: %s\n", recv_length, recv_msg );


        // Display received string onto console
        
    }while( strncmp(sent_msg, "bye", 3) );      // Loop until the sent message is "bye"
    
    printf ("Disconnection\n");
    close(udp_socket);
}

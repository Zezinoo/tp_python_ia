//


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
    int udp_socket = socket(AF_INET, SOCK_DGRAM, 17);
    if ( udp_socket < 0 ) {
        perror("\nERROR WHILE OPENING SOCKET:\n");
        exit(-1);
    }
    
    
    char sent_msg[MAX_LENGTH];
    char recv_msg[MAX_LENGTH];
    int  recv_length;
    printf("Sending message to server\n\n");
    // C array acquired in wireshark
    char msg[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x45, 0x00,
    0x00, 0x3c, 0x2c, 0xa2, 0x00, 0x00, 0x40, 0x11,
    0x4f, 0xd9, 0x7f, 0x00, 0x00, 0x01, 0x7f, 0x00,
    0x00, 0x35, 0xaf, 0x9c, 0x00, 0x35, 0x00, 0x28,
    0xfe, 0x6f, 0x59, 0xce, 0x01, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x6e,
    0x73, 0x08, 0x65, 0x6e, 0x73, 0x69, 0x63, 0x61,
    0x65, 0x6e, 0x02, 0x66, 0x72, 0x00, 0x00, 0x01,
    0x00, 0x01
    };


    int msg_length = sizeof(msg);
    // Send message to the server through socket. Needs an ip adress and a port

    sendto(udp_socket, msg, msg_length, 0, (struct sockaddr *) &server_address, address_length);

    printf("Message sent.\n\n");
    
    //printf("%d", msg_length);

    // Display received string onto console

    // Read up to MAX_LENGTH characters from server and put '\0' as last character to turn this array into a string
    // Trying to read the response from the DNS request
    recv_length = recvfrom(udp_socket, recv_msg, MAX_LENGTH, 0, (struct sockaddr *)&server_address,  &address_length);
    printf( "Received %d chars: %s\n", recv_length, recv_msg );

    printf ("Disconnection\n");
    close(udp_socket);
}

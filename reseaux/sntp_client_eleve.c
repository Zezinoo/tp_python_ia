/**
 * SNTP client. Arguments: server IP address (decimal-and-dots format)
 * 
 * Client that connects to a SNTP server with the given IP address
 * Display the content of the response and the corresponding times
 * 
 * @author  Dimitri Boudier,   ENSICAEN
 * @author  PhL
 * @date    25/11/2020
 */


#include <errno.h>                      // Error management
#include <sys/socket.h>                 // Socket
#include <arpa/inet.h>                  // Internet
#include <stdio.h>                      // I/O
#include <stdlib.h> 
#include <string.h> 
#include <unistd.h>                     // Removes warning of implicit declaration when closing socket
#include <netdb.h>						// DNS request

#define SNTP_PORT           123         // SNTP port number
#define SNTP_FRAME_LENGTH   48          // SNTP frame length

#define MAX_LENGTH 64    

/*  https://www.ietf.org/rfc/rfc2030.txt

      0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    0 |LI | VN  |Mode |    Stratum    |     Poll      |   Precision   |
      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    4 |                          Root Delay                           |
      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    8 |                       Root Dispersion                         |
      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   12 |                     Reference Identifier                      |
      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   16 |                                                               |
      |                   Reference Timestamp (64)                    |
      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   24 |                                                               |
      |                   Originate Timestamp (64)                    |
      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   32 |                                                               |
      |                    Receive Timestamp (64)                     |
      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   40 |                                                               |
      |                    Transmit Timestamp (64)                    |
      +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
*/

/**
 * Perform an IPv4 DNS request
 * It could seem tricky but we want only IPv4 response
 * @param hostName : the hostname we want the IP address
 * @return 0 if error or the searched ipv4 binary address.
 * */
uint32_t ipv4DnsRequest (const char *hostName) {
	struct addrinfo hints ; // to ask for IPv4 response and other stuff we do not need
    struct addrinfo *result ;	
    
     memset(&hints, 0, sizeof(struct addrinfo));
     hints.ai_family = AF_INET;    /* Allow IPv4  */
     int err = getaddrinfo(hostName, NULL, &hints, &result);
     if ( err != 0 ) {return 0 ; }
	// getaddrinfo() returns a list of address structures. We only return the first one !
	struct sockaddr_in *res_inaddr = (struct sockaddr_in *)(result->ai_addr) ; // cast struct sockaddr to struct sockaddr_in
	return res_inaddr->sin_addr.s_addr ;
}


int main( int argc, char **argv) {
    
    int    sock;                            // Connection socket
    int    sock_error;                      // Socket error management
    struct sockaddr_in server_address;      // Structure that will contain the server address
    
    socklen_t address_length = sizeof(struct sockaddr_in);  // Length of sockaddr_in structure

    
    // Check the number of arguments : "sntp_client complete_server_name"
    if (argc != 2 ) {
        printf ("ARGUMENT ERROR. USAGE: ./sntp_client complete_server_name \n");
        exit (-1);
    }
    // argv[1] contains the SNTP server's name, i.e. fr.pool.ntp.org
    
    
    // Set server parameters
    server_address.sin_family       = AF_INET;              // Server address is Internet-type
    server_address.sin_port         = htons (SNTP_PORT);    // Set server port number, hton converts unsigned short integer into network byte order 
    server_address.sin_addr.s_addr  = ipv4DnsRequest(argv[1]);   // Set server IP address 
    if (server_address.sin_addr.s_addr == 0 ) {
      printf("debug");
		perror ("pb dns request") ;
		exit(-1);
    }
    printf ("IP server adress of %s is %s\n", argv[1], inet_ntoa (server_address.sin_addr)) ;
    
    // Open socket in disconnected mode (UDP). 17 is UDP number ( cf. file /etc/protocols )
    /** TODO
     * WRITE HERE CODE TO OPEN UDP SOCKET
    
    */

   sock = socket(AF_INET, SOCK_DGRAM, 17);
    if ( sock < 0 ) {
        perror("\nERROR WHILE OPENING SOCKET:\n");
        exit(-1);
    }
    
        
    
    uint8_t request[SNTP_FRAME_LENGTH] = {0};   // String to be sent to server
    uint8_t reply  [SNTP_FRAME_LENGTH] = {0};   // String to be received from server
    uint8_t msg_length;                         // Length of received message
        
    // Prepare SNTP request frame with valid parameters:
    // First byte the "Options" field
    //     b7-6: LI = Leap Indicator = 0 (no warning)
    //     b5-3: VN = Version Number = 4 (IPv4, IPv6 and OSI)
    //     b2-0: Mode = 3 (client)
    request[0] = (0 << 6) | (4 << 3) | 3;
    
    
    // Send SNTP request
    
    /** TODO
     * WRITE HERE CODE TO SEND THE REQUEST PACKET
    
    */
    printf("Sending request : %s " , request);
    msg_length = strlen(request);
    sendto(sock, request, msg_length, 0, (struct sockaddr *) &server_address, address_length);
    
    // Read SNTP reply
   /** TODO
     * WRITE HERE CODE TO RECEIVE RESPONSE PACKET
    
    */

    uint8_t recv_length = recvfrom(sock, reply, SNTP_FRAME_LENGTH, 0, (struct sockaddr *)&server_address,  &address_length);
    

    
    // TODO : Display what has been received
    
    printf( "Received %d chars: %s\n", recv_length, reply );
    
    // TODO : Get timestamp from the integer part of the "Transmit Timestamp" field (bytes 40-43)
   
            
    // TODO : Print values in time format
  
    
    
    close(sock);                                // Close socket
}

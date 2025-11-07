# About

tcp_client_eleve_multi_client and tcp_server_multi_client implement a simple terminal based message forum. 

tcp_server_multi_client is the host end
tcp_client_eleve_multi_client is the client end 
Both implement tcp protocol connections with c wrappers for socket operation syscalls.

Up to 100 clients can connect to the host at a given time and send messages simultaneously. Their chosen name is displayed with each message.

# Usage

Compile and launch the server program with the port parameter

Compile and launch the client program with the ip and port parameter


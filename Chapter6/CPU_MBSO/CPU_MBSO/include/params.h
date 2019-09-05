#ifndef __PARAMS_H__
#define __PARAMS_H__

#include <string>
#include <stdlib.h>
#include <stdio.h>
#include "../include/error.h"
using namespace std;
typedef struct parsedconfigs_t {
	char ** configs;
	int lines;
} parsedconfigs;

extern "C"
//load the config from the input file
error parseConfig(char* filename, parsedconfigs** parsed_configs);
//delete the config
error freeConfig(parsedconfigs* parsed_configs);
//transfer the char array to real if the first word of this line is same with "string"
error getDouble(parsedconfigs* config, char* string, double* result);
//transfer the char array to int if the first word of this line is same with "string"
error getInt(parsedconfigs* config, char* string, int* result);
//transfer the char array to long if the first word of this line is same with "string
char* getParam(char* needle, char* haystack[], int count);

char* isParam(char * needle, char* haystack[], int count);

#endif

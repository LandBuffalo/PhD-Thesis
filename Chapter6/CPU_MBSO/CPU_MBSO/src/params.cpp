#include "../include/params.h"

error getDouble(parsedconfigs* config, char* string, double* result) {
	char** buffer = config->configs;
	int count = config->lines;
	int i = 0;
	for (i = 0; i < count; i++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				*result = (double)atof(item);
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}

error getInt(parsedconfigs* config, char* string, int* result) {
	char** buffer = config->configs;
	int count = config->lines;
	int i = 0;
	for (i = 0; i < count; i++) {
		if (strstr(buffer[i], string) != NULL) {
			char* item = strtok(buffer[i], " ");
			if (item == NULL) {
				return ERRORINVALIDPARAM;
			}
			if (strcmp(item, string) == 0) {
				item = strtok(NULL, " ");
				if (item == NULL) {
					return ERRORINVALIDPARAM;
				}
				*result = atoi(item);
				return SUCCESS;
			}
		}
	}
	return ERRORNOPARAM;
}

char* getParam(char * needle, char* haystack[], int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if (strcmp(needle, haystack[i]) == 0) {
			if (i < count - 1) {
				return haystack[i + 1];
			}
		}
	}
	return 0;
}


char* isParam(char* needle, char* haystack[], int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if (strcmp(needle, haystack[i]) == 0) {
			if (i < count - 1) {
				return haystack[i + 1];
			}
		}
	}
	return 0;
}


error parseConfig(char* filename, parsedconfigs** parsed_configs) {
	FILE* cfile = fopen(filename, "r");
	if (cfile == NULL) return ERRORINVALIDCONFIG;
	parsedconfigs * result = new parsedconfigs;
	char * buffer = new char[5000];
	int lines = 0;
	while (fgets(buffer, 5000, cfile) != NULL) lines++;
	rewind(cfile);

	char **configs = (char**)malloc(lines * sizeof(char*));
	char *current;
	int i = 0;
	for (i = 0; i < lines; i++) {
		configs[i] = NULL;
		current = fgets(buffer, 5000, cfile);
		if (current != NULL) {
			if (current[0] != '#') {
				configs[i] = (char*)malloc((strlen(current) + 1) * sizeof(char));
				strcpy(configs[i], current);
			}
		}
		else {
			configs[i] = NULL;
		}
	}

	result->configs = configs;
	result->lines = lines;
	free(buffer);
	fclose(cfile);
	*parsed_configs = result;
	return SUCCESS;
}

error freeConfig(parsedconfigs* config) {
	int i;
	for (i = 0; i < config->lines; i++) {
		if (config->configs[i] != NULL) {
			free(config->configs[i]);
		}
	}

	free(config->configs);
	free(config);
	return SUCCESS;
}

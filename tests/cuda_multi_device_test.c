#include <stdio.h>
#include <stdlib.h>
#include "ds4_gpu.h"

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    int devices[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int n = sizeof(devices) / sizeof(devices[0]);

    if (!ds4_gpu_init_multi(devices, n)) {
        fprintf(stderr, "ds4_gpu_init_multi failed\n");
        return 1;
    }

    fprintf(stderr, "device count: %d\n", ds4_gpu_device_count());

    const char *model_path = "/root/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open model\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    void *map = malloc(size);
    if (fread(map, 1, size, f) != (size_t)size) {
        fprintf(stderr, "failed to read model\n");
        return 1;
    }
    fclose(f);

    for (int i = 0; i < n; i++) {
        ds4_gpu_set_device(devices[i]);
        if (!ds4_gpu_set_model_map(map, size)) {
            fprintf(stderr, "ds4_gpu_set_model_map failed for device %d\n", devices[i]);
            return 1;
        }
        fprintf(stderr, "device %d model map ok\n", devices[i]);
    }

    ds4_gpu_cleanup();
    free(map);
    return 0;
}

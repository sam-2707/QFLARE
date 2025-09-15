# SGX Enclave Makefile for QFlare Federated Learning
# This Makefile builds the Intel SGX enclave for secure aggregation

# SGX SDK Configuration
SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 1

# Include paths
include $(SGX_SDK)/buildenv.mk

# Compiler settings
ifeq ($(shell getconf LONG_BIT), 32)
	SGX_ARCH := x86
else ifeq ($(findstring -m32, $(CXXFLAGS)), -m32)
	SGX_ARCH := x86
endif

ifeq ($(SGX_ARCH), x86)
	SGX_COMMON_CFLAGS := -m32
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x86/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x86/sgx_edger8r
else
	SGX_COMMON_CFLAGS := -m64
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib64
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r
endif

ifeq ($(SGX_DEBUG), 1)
ifeq ($(SGX_PRERELEASE), 1)
$(error Cannot set SGX_DEBUG and SGX_PRERELEASE at the same time!!)
endif
endif

ifeq ($(SGX_DEBUG), 1)
        SGX_COMMON_CFLAGS += -O0 -g
else
        SGX_COMMON_CFLAGS += -O2
endif

# Project settings
Project_Name := qflare_enclave
Enclave_Name := $(Project_Name).so
Signed_Enclave_Name := $(Project_Name).signed.so

# Enclave source files
Enclave_Cpp_Files := qflare_enclave.c
Enclave_Cpp_Objects := $(Enclave_Cpp_Files:.c=.o)

# EDL file and generated files
Enclave_EDL_File := qflare_enclave.edl
Enclave_Trusted_Include := qflare_enclave_t.h
Enclave_Trusted_Object := qflare_enclave_t.o

# Configuration files
Enclave_Config_File := qflare_enclave.config.xml
Enclave_Private_Key := qflare_enclave_private.pem

# Compiler flags
Enclave_C_Flags := $(SGX_COMMON_CFLAGS) -nostdinc -fvisibility=hidden -fpie -ffunction-sections -fdata-sections
Enclave_C_Flags += -Wall -Wextra -Winit-self -Wpointer-arith -Wreturn-type \
                   -Waddress -Wsequence-point -Wformat-security \
                   -Wmissing-include-dirs -Wfloat-equal -Wundef -Wshadow \
                   -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls

# Include directories
Enclave_Include_Paths := -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx -I.

# Linker flags
Enclave_Link_Flags := $(SGX_COMMON_CFLAGS) -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_LIBRARY_PATH) \
	-Wl,--whole-archive -lsgx_trts -Wl,--no-whole-archive \
	-Wl,--start-group -lsgx_tstdc -lsgx_tcrypto -lsgx_tservice -Wl,--end-group \
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
	-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections

# Targets
.PHONY: all clean

all: $(Signed_Enclave_Name)

# Generate trusted proxy functions from EDL
$(Enclave_Trusted_Include): $(Enclave_EDL_File)
	@$(SGX_EDGER8R) --trusted $(Enclave_EDL_File) --search-path $(SGX_SDK)/include
	@echo "GEN  =>  $@"

# Compile enclave source files
%.o: %.c $(Enclave_Trusted_Include)
	@$(CC) $(Enclave_C_Flags) $(Enclave_Include_Paths) -c $< -o $@
	@echo "CC   <=  $<"

# Compile trusted proxy functions
$(Enclave_Trusted_Object): $(Enclave_Trusted_Include)
	@$(CC) $(Enclave_C_Flags) $(Enclave_Include_Paths) -c qflare_enclave_t.c -o $@
	@echo "CC   <=  qflare_enclave_t.c"

# Link enclave
$(Enclave_Name): $(Enclave_Cpp_Objects) $(Enclave_Trusted_Object)
	@$(CC) $^ -o $@ $(Enclave_Link_Flags)
	@echo "LINK =>  $@"

# Generate private key if it doesn't exist
$(Enclave_Private_Key):
	@openssl genrsa -out $@ -3 3072
	@echo "Generated enclave private key: $@"
	@echo "WARNING: This is a development key. Use proper key management in production!"

# Sign enclave
$(Signed_Enclave_Name): $(Enclave_Name) $(Enclave_Private_Key)
	@$(SGX_ENCLAVE_SIGNER) sign -key $(Enclave_Private_Key) -enclave $(Enclave_Name) -out $@ -config $(Enclave_Config_File)
	@echo "SIGN =>  $@"

# Install target
install: $(Signed_Enclave_Name)
	@mkdir -p ../server/enclave/sgx/
	@cp $(Signed_Enclave_Name) ../server/enclave/sgx/
	@cp $(Enclave_Trusted_Include) ../server/enclave/sgx/
	@echo "Installed enclave to ../server/enclave/sgx/"

# Clean build artifacts
clean:
	@rm -f $(Enclave_Name) $(Signed_Enclave_Name) $(Enclave_Cpp_Objects) $(Enclave_Trusted_Object)
	@rm -f qflare_enclave_t.* qflare_enclave_u.*
	@echo "Cleaned build artifacts"

# Development targets
dev-key: $(Enclave_Private_Key)

check-sgx:
	@echo "Checking SGX environment..."
	@if [ ! -d "$(SGX_SDK)" ]; then \
		echo "ERROR: SGX SDK not found at $(SGX_SDK)"; \
		echo "Please install Intel SGX SDK and set SGX_SDK environment variable"; \
		exit 1; \
	fi
	@if [ ! -f "/dev/sgx_enclave" ] && [ ! -f "/dev/isgx" ]; then \
		echo "WARNING: SGX device not found. Running in simulation mode."; \
		echo "Install SGX driver for hardware mode."; \
	fi
	@echo "SGX environment check completed"

debug: SGX_DEBUG=1
debug: all

release: SGX_DEBUG=0
release: clean all

# Help target
help:
	@echo "QFlare SGX Enclave Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build and sign the enclave"
	@echo "  clean       - Remove build artifacts"
	@echo "  install     - Install enclave to server directory"
	@echo "  dev-key     - Generate development private key"
	@echo "  check-sgx   - Check SGX environment"
	@echo "  debug       - Build with debug symbols"
	@echo "  release     - Build optimized release version"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Environment Variables:"
	@echo "  SGX_SDK     - Path to Intel SGX SDK (default: /opt/intel/sgxsdk)"
	@echo "  SGX_MODE    - Build mode: HW or SIM (default: HW)"
	@echo "  SGX_DEBUG   - Debug build: 1 or 0 (default: 1)"
	@echo "  SGX_ARCH    - Architecture: x64 or x86 (auto-detected)"
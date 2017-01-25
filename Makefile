MYDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

CD ?= cd
CP ?= cp
CMAKE ?= cmake
MAKE ?= make
MKDIR ?= mkdir
OBJCOPY ?= objcopy
RM ?= rm
STRIP ?= strip

BUILD_DIR ?= $(MYDIR)/build
NATIVE_DIR := $(BUILD_DIR)/bm

SRC_DIR := $(MYDIR)/src
SRC_CONFIG_FLAGS ?=

ifndef CMAKE_PREFIX_PATH
$(error CMAKE_PREFIX_PATH is undefined)
endif

all: hosted native

clean:
	-$(RM) -r $(BUILD_DIR)

hosted: $(BUILD_DIR)/Makefile
	$(MAKE) -C $(BUILD_DIR) 

native: $(NATIVE_DIR) | $(NATIVE_DIR)/reconstruction.elf32

$(BUILD_DIR)/Makefile: | $(BUILD_DIR)
	$(CD) $(BUILD_DIR) && $(CMAKE) \
	-DCMAKE_PREFIX_PATH=$(CMAKE_PREFIX_PATH) \
	-DCMAKE_BUILD_TYPE=Release $(MYDIR)
#	-DCMAKE_BUILD_TYPE=Debug $(MYDIR)

$(BUILD_DIR):
	$(MKDIR) $@

$(NATIVE_DIR): | $(BUILD_DIR)
	$(MKDIR) $@

check-env:
ifndef EBBRT_SYSROOT
	$(error EBBRT_SYSROOT is undefined)
endif

$(SRC_DIR): check-env
	$(CD) $(NATIVE_DIR) && $(CMAKE) \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_TOOLCHAIN_FILE=$(EBBRT_SYSROOT)/usr/misc/ebbrt.cmake ../../

$(NATIVE_DIR)/reconstruction.elf: $(SRC_DIR) | $(NATIVE_DIR)
	$(MAKE) -C $(NATIVE_DIR)

%.elf.stripped: %.elf
	$(STRIP) -s $< -o $@

%.elf32: %.elf.stripped
	$(OBJCOPY) -O elf32-i386 $< $@

.PHONY: all check-env clean hosted native

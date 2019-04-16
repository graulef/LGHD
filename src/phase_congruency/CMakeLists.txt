CMAKE_MINIMUM_REQUIRED(VERSION 3.6)

PROJECT("phase_congruency")

# -------------------------------------------------------------------------------------------------

FIND_PACKAGE(ITK REQUIRED)
IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
ENDIF(ITK_FOUND)

# -------------------------------------------------------------------------------------------------

SET(LIBRARY_NAME
    "bip_phase_congruency"
)
SET(LIBRARY_SOURCE_FILES
    "phase_congruency.cpp"
)
ADD_LIBRARY(
    ${LIBRARY_NAME}
    ${LIBRARY_SOURCE_FILES}
)
TARGET_LINK_LIBRARIES(
    ${LIBRARY_NAME}
    bip_utils
    bip_filter_bank
)
TARGET_INCLUDE_DIRECTORIES(
    ${LIBRARY_NAME}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}
)
SET_PROPERTY(
    TARGET ${LIBRARY_NAME}
    PROPERTY CXX_STANDARD 14)
SET_PROPERTY(
    TARGET ${LIBRARY_NAME}
    PROPERTY CXX_STANDARD_REQUIRED ON)


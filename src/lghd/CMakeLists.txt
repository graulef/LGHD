CMAKE_MINIMUM_REQUIRED(VERSION 3.6)

PROJECT("examples")

find_package( OpenCV REQUIRED )
if( OpenCV_FOUND )
    list( APPEND ThirdParty_LIBS ${OpenCV_LIBS} )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
endif( OpenCV_FOUND )


SET(PROGRAM_NAME
        "lghd"
        )
SET(PROGRAM_SOURCE_FILES
        "lghd.cpp"
        )
ADD_EXECUTABLE(
        ${PROGRAM_NAME}
        ${PROGRAM_SOURCE_FILES}
)
TARGET_LINK_LIBRARIES(
        ${PROGRAM_NAME}
        bip_utils
        bip_filter_bank
        bip_phase_congruency
        ${OpenCV_LIBS}
)
SET_PROPERTY(
        TARGET ${PROGRAM_NAME}
        PROPERTY CXX_STANDARD 14)
SET_PROPERTY(
        TARGET ${PROGRAM_NAME}
        PROPERTY CXX_STANDARD_REQUIRED ON)

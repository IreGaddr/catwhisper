find_program(GLSLANG_VALIDATOR glslangValidator)

if(NOT GLSLANG_VALIDATOR)
    message(WARNING "glslangValidator not found, shaders won't be compiled at build time")
endif()

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

set(SHADER_SOURCES
    distance_l2.comp
    distance_ip.comp
    topk_heap.comp
)

set(SHADER_OUTPUTS "")

foreach(SHADER_NAME ${SHADER_SOURCES})
    set(SHADER_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_NAME}")
    set(OUTPUT_PATH "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.spv")
    
    if(GLSLANG_VALIDATOR)
        add_custom_command(
            OUTPUT ${OUTPUT_PATH}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
            COMMAND ${GLSLANG_VALIDATOR} -V --target-env spirv1.3 ${SHADER_PATH} -o ${OUTPUT_PATH}
            DEPENDS ${SHADER_PATH}
            COMMENT "Compiling shader ${SHADER_NAME}"
        )
        list(APPEND SHADER_OUTPUTS ${OUTPUT_PATH})
    endif()
endforeach()

if(SHADER_OUTPUTS)
    add_custom_target(catwhisper_shaders ALL DEPENDS ${SHADER_OUTPUTS})
    add_dependencies(catwhisper catwhisper_shaders)
endif()

install(FILES ${SHADER_OUTPUTS}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/catwhisper/shaders
)

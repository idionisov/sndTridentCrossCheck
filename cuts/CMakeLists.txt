# Header files for dictionary generation
set(CUTS_HEADERS
  sndBaseCut.h
  sndSciFiBaseCut.h
  sndMuFilterBaseCut.h
  sndMinSciFiHitsCut.h
  sndSciFiStationCut.h
  sndVetoCut.h
  sndHasVetoHitsCut.h
  sndMinSciFiConsecutivePlanes.h
  sndDSActivityCut.h
  sndUSQDCCut.h
  sndEventHeaderBaseCut.h
  sndEventDeltat.h
  sndAvgSciFiFiducialCut.h
  sndAvgDSFiducialCut.h
  sndDSVetoCut.h
  sndMinSciFiPlanesCut.h
  sndTridentHitsCut.h
  sndMaxSciFiHitsCut.h
  sndMaxPlaneSciFiHitsCut.h
  sndMaxSciFiSignalCut.h
  sndMaxPlaneSciFiSignalCut.h
  sndTridentDensityCut.h
)

set(CUTS_SRCS
  sndSciFiBaseCut.cxx
  sndMuFilterBaseCut.cxx
  sndMinSciFiHitsCut.cxx
  sndSciFiStationCut.cxx
  sndVetoCut.cxx
  sndHasVetoHitsCut.cxx
  sndMinSciFiConsecutivePlanes.cxx
  sndDSActivityCut.cxx
  sndUSQDCCut.cxx
  sndEventHeaderBaseCut.cxx
  sndEventDeltat.cxx
  sndAvgSciFiFiducialCut.cxx
  sndAvgDSFiducialCut.cxx
  sndDSVetoCut.cxx
  sndMinSciFiPlanesCut.cxx
  sndTridentHitsCut.cxx
  sndMaxSciFiHitsCut.cxx
  sndMaxPlaneSciFiHitsCut.cxx
  sndMaxSciFiSignalCut.cxx
  sndMaxPlaneSciFiSignalCut.cxx
  sndTridentDensityCut.cxx
)

# ROOT generate dictionary
root_generate_dictionary(G__trident_cuts
  ${CUTS_HEADERS}
  LINKDEF AnalysisCutsLinkDef.h
)

# Add library for cuts
add_library(trident_cuts SHARED
  ${CUTS_SRCS}
  G__trident_cuts.cxx
)

# Find sndsw dependencies
find_library(SHIP_LHC_LIB shipLHC PATHS "${SNDSW_ROOT}/lib" REQUIRED)
find_library(VETO_LIB veto PATHS "${SNDSW_ROOT}/lib" REQUIRED)
find_library(ANALYSIS_TOOLS_LIB snd_analysis_tools PATHS "${SNDSW_ROOT}/lib" REQUIRED)

target_link_libraries(trident_cuts
  PUBLIC
  ${ROOT_LIBRARIES}
  ${SHIP_LHC_LIB}
  ${VETO_LIB}
  ${ANALYSIS_TOOLS_LIB}
)

# Set library properties
set_target_properties(trident_cuts PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

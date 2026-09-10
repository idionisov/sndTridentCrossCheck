#pragma once

#include "TClonesArray.h"
#include "ShipMCTrack.h"
#include "TMCProcess.h"
#include <cmath>
#include <string>
#include <optional>
#include <utility>

namespace snd::trident {

enum ProcessType {
    kUnknown = -1,
    kMuonToMuonPair = 0, // Genuine Trident: mu -> 3mu
    kGammaToMuPair = 1,  // Photon conversion: gamma -> 2mu
    kAnnihiToMuPair = 2  // Positron annihilation: e+ e- -> 2mu
};

enum RegionType {
    kRegionUnknown = 0,
    kRegionRock = 1,       // Vertex in upstream rock (Z < targetZMin)
    kRegionTarget = 2,     // Vertex in target (targetZMin <= Z < targetZMax)
    kRegionMuonSystem = 3  // Vertex downstream in muon system (Z >= targetZMax)
};

enum FlagBits {
    kIsGenuine        = 1 << 0,  // Genuine Trident
    kIsGammaConv      = 1 << 1,  // Photon conversion
    kIsPositronAnn    = 1 << 2,  // Positron annihilation
    kIsPrimaryMuon    = 1 << 3,  // Radiating muon is primary beam
    kIsSecondaryMuon  = 1 << 4,  // Radiating muon produced in shower
    kIsDirectBrem     = 1 << 5,  // Photon emitted directly by muon
    kInRock           = 1 << 6,  // Vertex in rock
    kInTarget         = 1 << 7,  // Vertex in target
    kInMuonSystem     = 1 << 8,  // Vertex in muon system
    kInFiducial       = 1 << 9   // All 3 muons crossed fiducial area
};

struct TridentTruthInfo {
    // Candidate status
    bool hasCandidate{false}; // True if an opposite-sign muon pair was identified

    // Categorization
    ProcessType procType{kUnknown};
    RegionType regionType{kRegionUnknown};
    unsigned int tridentFlags{0};
    bool isFiducial{false};

    // Event Weights
    double mcWeight{0.0};
    double scaledWeight{0.0};

    // Interaction vertex (cm)
    double vtxX{0.0};
    double vtxY{0.0};
    double vtxZ{-9999.0};

    // Track indices in MCTrack collection
    int radMuonId{-1};
    int muMinusId{-1};
    int muPlusId{-1};

    // Kinematics - Radiating/Incoming Muon
    double pMuIn{0.0};
    double pxMuIn{0.0};
    double pyMuIn{0.0};
    double pzMuIn{0.0};
    double ptMuIn{0.0};

    // Kinematics - mu-
    double pMuMinus{0.0};
    double pxMuMinus{0.0};
    double pyMuMinus{0.0};
    double pzMuMinus{0.0};
    double ptMuMinus{0.0};
    double etaMuMinus{0.0};
    double phiMuMinus{0.0};

    // Kinematics - mu+
    double pMuPlus{0.0};
    double pxMuPlus{0.0};
    double pyMuPlus{0.0};
    double pzMuPlus{0.0};
    double ptMuPlus{0.0};
    double etaMuPlus{0.0};
    double phiMuPlus{0.0};

    // Pair composite kinematics
    double invMass2Mu{0.0};        // Invariant mass of mu+ mu- pair [GeV/c^2]
    double openingAngleMrad{0.0};  // Opening angle between mu+ and mu- [mrad]
    double pt2Mu{0.0};             // Pair transverse momentum [GeV/c]
    double energyAsym{0.0};        // Energy asymmetry (E+ - E-) / (E+ + E-)

    // Angular separations
    double deltaPhi2Mu{0.0};       // Azimuthal angle separation |phi+ - phi-| folded into [0, pi] [rad]
    double deltaEta2Mu{0.0};       // Pseudorapidity separation |eta+ - eta-|
    double deltaR2Mu{0.0};         // Delta R = sqrt(deltaEta^2 + deltaPhi^2)

    // Tri-muon composite kinematics (radiating muon + mu+ mu- pair)
    double invMass3Mu{0.0};        // Total invariant mass of 3-muon system [GeV/c^2]
    double pt3Mu{0.0};             // Total transverse momentum of 3-muon system [GeV/c]
    double openingAngleRadMinusMrad{0.0}; // Angle between radiating muon and mu- [mrad]
    double openingAngleRadPlusMrad{0.0};  // Angle between radiating muon and mu+ [mrad]

    // Fractional & relative kinematics
    double energyRatioPair{0.0};         // (E(mu+) + E(mu-)) / E(mu_in)
    double momentumFractionMinus{0.0};   // p(mu-) / p(mu_in)
    double momentumFractionPlus{0.0};    // p(mu+) / p(mu_in)
    double ptAsymmetry{0.0};             // (pt(mu+) - pt(mu-)) / (pt(mu+) + pt(mu-))

    // Ancestry & shower detail
    int cascadeDepth{-1};
    TMCProcess emissionProc{kPNoProcess};

    const char* getProcessName() const {
        switch (procType) {
            case kMuonToMuonPair: return "Genuine";
            case kGammaToMuPair:  return "GammaConversion";
            case kAnnihiToMuPair: return "PositronAnnihilation";
            default:              return "Unknown";
        }
    }

    const char* getRegionName() const {
        switch (regionType) {
            case kRegionRock:       return "Rock";
            case kRegionTarget:     return "Target";
            case kRegionMuonSystem: return "MuonSystem";
            default:                return "Unknown";
        }
    }

    const char* getEmissionProcessName() const {
        int proc = static_cast<int>(emissionProc);
        if (proc >= 0 && proc < kMaxMCProcess) {
            return TMCProcessName[proc];
        }
        return "Unknown";
    }

    // Extrapolated transverse positions at fiducial plane Z = fidZ (cm)
    double fidXMinus{-9999.0};
    double fidYMinus{-9999.0};
    double fidXPlus{-9999.0};
    double fidYPlus{-9999.0};
    double fidXRad{-9999.0};
    double fidYRad{-9999.0};
    double distToFiducialEdge{-9999.0}; // Min distance to fiducial boundary for 3 muons (>0 inside, <0 outside) [cm]

    // --- Query Predicates for RDataFrame & Analysis Scripts ---
    bool isGenuineTrident() const { return procType == kMuonToMuonPair; }
    bool isGammaConversion() const { return procType == kGammaToMuPair; }
    bool isPositronAnnihilation() const { return procType == kAnnihiToMuPair; }

    bool inRock() const { return regionType == kRegionRock; }
    bool inTarget() const { return regionType == kRegionTarget; }
    bool inMuonSystem() const { return regionType == kRegionMuonSystem; }

    bool isPrimaryMuon() const { return (tridentFlags & kIsPrimaryMuon) != 0; }
    bool isSecondaryMuon() const { return (tridentFlags & kIsSecondaryMuon) != 0; }
    bool isDirectBrem() const { return (tridentFlags & kIsDirectBrem) != 0; }

    // Convenient composite queries
    bool isRockTrident() const { return inRock() && isGenuineTrident(); }
    bool isTargetTrident() const { return inTarget() && isGenuineTrident(); }
    bool isRockTriMuon() const { return hasCandidate && inRock(); }
    bool isTargetTriMuon() const { return hasCandidate && inTarget(); }

    bool matches(ProcessType proc, RegionType reg) const {
        return procType == proc && regionType == reg;
    }
};

struct TridentTruthConfig {
    double zMin{-1000.0};
    double zMax{1000.0};
    double targetZMin{-100.0};
    double targetZMax{600.0};
    int processMask{7};        // Bitmask: 1=genuine (1<<0), 2=gamma (1<<1), 4=annihil (1<<2)
    double weightScale{1.0};   // Overall normalization scale

    bool useFiducial{false};
    double fidZ{300.0};
    double fidXMin{-50.0};
    double fidXMax{50.0};
    double fidYMin{-50.0};
    double fidYMax{50.0};
};

struct AncestryResult {
    ProcessType procType{kUnknown};
    int radMuonId{-1};
    int cascadeDepth{0};
    TMCProcess emissionProc{kPNoProcess};
    bool isPrimaryMuon{false};
    bool isDirectBrem{false};
};

struct FiducialResult {
    bool isFiducial{false};
    double xMinus{-9999.0};
    double yMinus{-9999.0};
    double xPlus{-9999.0};
    double yPlus{-9999.0};
    double xRad{-9999.0};
    double yRad{-9999.0};
    double distToEdge{-9999.0};
};

class TridentTruthProcessor {
    public:
        TridentTruthProcessor() = default;
        explicit TridentTruthProcessor(const TridentTruthConfig& config) : fConfig(config) {}

        void setConfig(const TridentTruthConfig& config) { fConfig = config; }
        const TridentTruthConfig& getConfig() const { return fConfig; }

        // Evaluates truth tracks and returns structured result
        TridentTruthInfo process(const TClonesArray& mcTracks) const;

        // Callable interface for RDataFrame Define: df.Define("truth", processor, ["MCTrack"])
        TridentTruthInfo operator()(const TClonesArray& mcTracks) const {
            return process(mcTracks);
        }

        static bool checkMuonFiducial(const ShipMCTrack* mcTrk, double zPlane,
                                    double xMin, double xMax, double yMin, double yMax,
                                    double& xOut, double& yOut);

    private:
        TridentTruthConfig fConfig;

        // Modular sub-task helpers
        std::optional<std::pair<int, int>> findCandidatePair(const TClonesArray& mcTracks) const;
        AncestryResult traceAncestry(const TClonesArray& mcTracks, int motherId, int procId) const;
        RegionType determineRegion(double zVertex) const;
        const ShipMCTrack* getRadiatingTrack(const TClonesArray& mcTracks, int radMuonId) const;
        FiducialResult checkFiducialTriple(const ShipMCTrack* mcTrkMinus, const ShipMCTrack* mcTrkPlus, const ShipMCTrack* radMcTrk) const;
        void fillKinematics(TridentTruthInfo& info, const ShipMCTrack* mcTrkMinus, const ShipMCTrack* mcTrkPlus, const ShipMCTrack* radMcTrk) const;
        unsigned int packFlags(const AncestryResult& ancestry, RegionType region, bool isFiducial) const;
        double calculateWeight(double rawWeight) const;
        TridentTruthInfo makeEmptyInfo(const TClonesArray& mcTracks) const;
};

} // namespace snd::trident

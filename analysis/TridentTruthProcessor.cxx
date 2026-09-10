#include "TridentTruthProcessor.h"
#include <algorithm>
#include <cmath>

namespace snd::trident {

bool TridentTruthProcessor::checkMuonFiducial(
    const ShipMCTrack* mcTrk,
    double zPlane,
    double xMin, double xMax, double yMin, double yMax,
    double& xOut, double& yOut
)
{
    if (!mcTrk) {
        xOut = -9999.0; yOut = -9999.0;
        return false;
    }
    double zStart = mcTrk->GetStartZ();
    double pz = mcTrk->GetPz();
    if (pz <= 0.0) {
        xOut = mcTrk->GetStartX();
        yOut = mcTrk->GetStartY();
        return false;
    }
    double deltaZ = zPlane - zStart;
    xOut = mcTrk->GetStartX() + (mcTrk->GetPx() / pz) * deltaZ;
    yOut = mcTrk->GetStartY() + (mcTrk->GetPy() / pz) * deltaZ;

    return (xOut >= xMin && xOut <= xMax && yOut >= yMin && yOut <= yMax);
}

std::optional<std::pair<int, int>> TridentTruthProcessor::findCandidatePair(const TClonesArray& mcTracks) const
{
    int nTracks = mcTracks.GetEntries();
    for (int i_mcTrk = 0; i_mcTrk < nTracks; ++i_mcTrk) {
        auto* mcTrk1 = static_cast<ShipMCTrack*>(mcTracks.At(i_mcTrk));
        if (!mcTrk1) continue;

        int procId1 = mcTrk1->GetProcID();
        if (procId1 != kPPair && procId1 != kPAnnihilation && procId1 != kPAnnihilationRest && procId1 != kPAnnihilationFlight) continue;
        int pdg1 = mcTrk1->GetPdgCode();
        if (std::abs(pdg1) != 13) continue;

        for (int j_mcTrk = i_mcTrk + 1; j_mcTrk < nTracks; ++j_mcTrk) {
            auto* mcTrk2 = static_cast<ShipMCTrack*>(mcTracks.At(j_mcTrk));
            if (!mcTrk2) continue;

            if (mcTrk2->GetProcID() == procId1 && mcTrk2->GetMotherId() == mcTrk1->GetMotherId())
            {
                int pdg2 = mcTrk2->GetPdgCode();
                if ((pdg1 == 13 && pdg2 == -13) || (pdg1 == -13 && pdg2 == 13)) {
                    double zStart1 = mcTrk1->GetStartZ();
                    if (zStart1 >= fConfig.zMin && zStart1 < fConfig.zMax) {
                        int minusId = (pdg1 == 13) ? i_mcTrk : j_mcTrk;
                        int plusId  = (pdg1 == -13) ? i_mcTrk : j_mcTrk;
                        return std::make_pair(minusId, plusId);
                    }
                }
            }
        }
    }
    return std::nullopt;
}

AncestryResult TridentTruthProcessor::traceAncestry(
    const TClonesArray& mcTracks,
    int motherId, int procId
) const
{
    AncestryResult res;
    res.emissionProc = static_cast<TMCProcess>(procId);
    int nTracks = mcTracks.GetEntries();

    if (motherId == -1) {
        res.procType = kMuonToMuonPair;
        res.isPrimaryMuon = true;
        res.isDirectBrem = true;
        res.emissionProc = kPPair;
        return res;
    }

    if (motherId >= 0 && motherId < nTracks)
    {
        auto* motherMcTrk = static_cast<ShipMCTrack*>(mcTracks.At(motherId));
        if (!motherMcTrk) return res;

        int motherPdg = motherMcTrk->GetPdgCode();
        if (std::abs(motherPdg) == 13) {
            res.procType = kMuonToMuonPair;
            res.radMuonId = motherId;
            res.isPrimaryMuon = (motherMcTrk->GetMotherId() == -1);
            res.isDirectBrem = true;
            res.emissionProc = kPPair;
        }
        else if (motherPdg == 22 || std::abs(motherPdg) == 11)
        {
            res.procType = (motherPdg == 22) ? kGammaToMuPair : kAnnihiToMuPair;

            int currentTrackId = motherId;
            int childOfMuonTrackId = -1;
            while (currentTrackId >= 0 && currentTrackId < nTracks)
            {
                auto* ancestorMcTrk = static_cast<ShipMCTrack*>(mcTracks.At(currentTrackId));
                if (!ancestorMcTrk) break;

                res.cascadeDepth++;
                if (std::abs(ancestorMcTrk->GetPdgCode()) == 13)
                {
                    res.radMuonId = currentTrackId;
                    res.isPrimaryMuon = (ancestorMcTrk->GetMotherId() == -1);
                    res.cascadeDepth--;
                    break;
                }
                childOfMuonTrackId = currentTrackId;
                currentTrackId = ancestorMcTrk->GetMotherId();
            }

            if (childOfMuonTrackId >= 0 && childOfMuonTrackId < nTracks)
            {
                auto* childMcTrk = static_cast<ShipMCTrack*>(mcTracks.At(childOfMuonTrackId));
                if (childMcTrk) res.emissionProc = static_cast<TMCProcess>(childMcTrk->GetProcID());
            }
            res.isDirectBrem = (motherPdg == 22 && motherMcTrk->GetMotherId() == res.radMuonId && res.cascadeDepth == 1);
        }
    }
    return res;
}

RegionType TridentTruthProcessor::determineRegion(double zVertex) const
{
    if (zVertex < fConfig.targetZMin) return kRegionRock;
    if (zVertex < fConfig.targetZMax) return kRegionTarget;
    return kRegionMuonSystem;
}

const ShipMCTrack* TridentTruthProcessor::getRadiatingTrack(
    const TClonesArray& mcTracks, int radMuonId
) const
{
    int nTracks = mcTracks.GetEntriesFast();
    if (radMuonId >= 0 && radMuonId < nTracks) {
        return static_cast<ShipMCTrack*>(mcTracks.At(radMuonId));
    }
    if (nTracks > 0) {
        return static_cast<ShipMCTrack*>(mcTracks.At(0));
    }
    return nullptr;
}

FiducialResult TridentTruthProcessor::checkFiducialTriple(
    const ShipMCTrack* mcTrkMinus,
    const ShipMCTrack* mcTrkPlus,
    const ShipMCTrack* radMcTrk
) const
{
    FiducialResult fid;
    bool fidMinus = checkMuonFiducial(mcTrkMinus, fConfig.fidZ, fConfig.fidXMin, fConfig.fidXMax, fConfig.fidYMin, fConfig.fidYMax, fid.xMinus, fid.yMinus);
    bool fidPlus  = checkMuonFiducial(mcTrkPlus,  fConfig.fidZ, fConfig.fidXMin, fConfig.fidXMax, fConfig.fidYMin, fConfig.fidYMax, fid.xPlus,  fid.yPlus);
    bool fidRad   = checkMuonFiducial(radMcTrk,   fConfig.fidZ, fConfig.fidXMin, fConfig.fidXMax, fConfig.fidYMin, fConfig.fidYMax, fid.xRad,   fid.yRad);

    fid.isFiducial = (fidMinus && fidPlus && fidRad);

    auto distEdge = [](double x, double y, double xMin, double xMax, double yMin, double yMax) {
        double dx = std::min(x - xMin, xMax - x);
        double dy = std::min(y - yMin, yMax - y);
        return std::min(dx, dy);
    };

    if (fid.xMinus > -9000.0 && fid.xPlus > -9000.0 && fid.xRad > -9000.0) {
        double dM = distEdge(fid.xMinus, fid.yMinus, fConfig.fidXMin, fConfig.fidXMax, fConfig.fidYMin, fConfig.fidYMax);
        double dP = distEdge(fid.xPlus, fid.yPlus, fConfig.fidXMin, fConfig.fidXMax, fConfig.fidYMin, fConfig.fidYMax);
        double dR = distEdge(fid.xRad, fid.yRad, fConfig.fidXMin, fConfig.fidXMax, fConfig.fidYMin, fConfig.fidYMax);
        fid.distToEdge = std::min({dM, dP, dR});
    }

    return fid;
}

unsigned int TridentTruthProcessor::packFlags(
    const AncestryResult& ancestry,
    RegionType region,
    bool isFiducial
) const
{
    unsigned int flags = 0;
    if (ancestry.procType == kMuonToMuonPair) flags |= kIsGenuine;
    else if (ancestry.procType == kGammaToMuPair) flags |= kIsGammaConv;
    else if (ancestry.procType == kAnnihiToMuPair) flags |= kIsPositronAnn;

    if (ancestry.isPrimaryMuon) flags |= kIsPrimaryMuon;
    else flags |= kIsSecondaryMuon;

    if (ancestry.isDirectBrem) flags |= kIsDirectBrem;

    if (region == kRegionRock) flags |= kInRock;
    else if (region == kRegionTarget) flags |= kInTarget;
    else if (region == kRegionMuonSystem) flags |= kInMuonSystem;

    if (isFiducial) flags |= kInFiducial;
    return flags;
}

double TridentTruthProcessor::calculateWeight(double rawWeight) const
{
    return rawWeight * fConfig.weightScale;
}

void TridentTruthProcessor::fillKinematics(
    TridentTruthInfo& info,
    const ShipMCTrack* mcTrkMinus,
    const ShipMCTrack* mcTrkPlus,
    const ShipMCTrack* radMcTrk
) const
{
    const double mMu = 0.1056583755;

    // incoming radiating muon
    if (radMcTrk) {
        info.pMuIn  = radMcTrk->GetP();
        info.pxMuIn = radMcTrk->GetPx();
        info.pyMuIn = radMcTrk->GetPy();
        info.pzMuIn = radMcTrk->GetPz();
        info.ptMuIn = std::sqrt(info.pxMuIn * info.pxMuIn + info.pyMuIn * info.pyMuIn);
    }

    // mu- kinematics
    info.pxMuMinus = mcTrkMinus->GetPx();
    info.pyMuMinus = mcTrkMinus->GetPy();
    info.pzMuMinus = mcTrkMinus->GetPz();
    info.pMuMinus  = mcTrkMinus->GetP();
    info.ptMuMinus = std::sqrt(info.pxMuMinus * info.pxMuMinus + info.pyMuMinus * info.pyMuMinus);
    info.phiMuMinus = std::atan2(info.pyMuMinus, info.pxMuMinus);
    if (info.pMuMinus - info.pzMuMinus > 1e-9 && info.pMuMinus + info.pzMuMinus > 1e-9) {
        info.etaMuMinus = 0.5 * std::log((info.pMuMinus + info.pzMuMinus) / (info.pMuMinus - info.pzMuMinus));
    }

    // mu+ kinematics
    info.pxMuPlus = mcTrkPlus->GetPx();
    info.pyMuPlus = mcTrkPlus->GetPy();
    info.pzMuPlus = mcTrkPlus->GetPz();
    info.pMuPlus  = mcTrkPlus->GetP();
    info.ptMuPlus = std::sqrt(info.pxMuPlus * info.pxMuPlus + info.pyMuPlus * info.pyMuPlus);
    info.phiMuPlus = std::atan2(info.pyMuPlus, info.pxMuPlus);
    if (info.pMuPlus - info.pzMuPlus > 1e-9 && info.pMuPlus + info.pzMuPlus > 1e-9) {
        info.etaMuPlus = 0.5 * std::log((info.pMuPlus + info.pzMuPlus) / (info.pMuPlus - info.pzMuPlus));
    }

    // Pair composite kinematics
    double eMinus = std::sqrt(info.pMuMinus * info.pMuMinus + mMu * mMu);
    double ePlus  = std::sqrt(info.pMuPlus * info.pMuPlus + mMu * mMu);
    double eTot = eMinus + ePlus;
    double pxTot = info.pxMuMinus + info.pxMuPlus;
    double pyTot = info.pyMuMinus + info.pyMuPlus;
    double pzTot = info.pzMuMinus + info.pzMuPlus;
    double pTot2 = pxTot * pxTot + pyTot * pyTot + pzTot * pzTot;
    double m2 = eTot * eTot - pTot2;

    info.invMass2Mu = (m2 > 0.0) ? std::sqrt(m2) : 0.0;
    info.pt2Mu = std::sqrt(pxTot * pxTot + pyTot * pyTot);
    info.energyAsym = (eTot > 0.0) ? (ePlus - eMinus) / eTot : 0.0;

    double pProd = info.pMuMinus * info.pMuPlus;
    double cosTheta = (pProd > 0.0) ? (info.pxMuMinus * info.pxMuPlus + info.pyMuMinus * info.pyMuPlus + info.pzMuMinus * info.pzMuPlus) / pProd : 1.0;
    cosTheta = std::clamp(cosTheta, -1.0, 1.0);
    info.openingAngleMrad = std::acos(cosTheta) * 1000.0;

    // Angular separations
    double dPhi = std::abs(info.phiMuPlus - info.phiMuMinus);
    const double pi = 3.14159265358979323846;
    if (dPhi > pi) dPhi = 2.0 * pi - dPhi;
    info.deltaPhi2Mu = dPhi;
    info.deltaEta2Mu = std::abs(info.etaMuPlus - info.etaMuMinus);
    info.deltaR2Mu   = std::sqrt(info.deltaEta2Mu * info.deltaEta2Mu + dPhi * dPhi);

    // Tri-muon kinematics & fractional ratios
    if (radMcTrk) {
        double eRad = std::sqrt(info.pMuIn * info.pMuIn + mMu * mMu);
        double e3Mu = eTot + eRad;
        double px3Mu = pxTot + info.pxMuIn;
        double py3Mu = pyTot + info.pyMuIn;
        double pz3Mu = pzTot + info.pzMuIn;
        double p3Mu2 = px3Mu * px3Mu + py3Mu * py3Mu + pz3Mu * pz3Mu;
        double m3Mu2 = e3Mu * e3Mu - p3Mu2;
        info.invMass3Mu = (m3Mu2 > 0.0) ? std::sqrt(m3Mu2) : 0.0;
        info.pt3Mu = std::sqrt(px3Mu * px3Mu + py3Mu * py3Mu);

        double pProdRadMinus = info.pMuIn * info.pMuMinus;
        if (pProdRadMinus > 0.0) {
            double cosTh = (info.pxMuIn * info.pxMuMinus + info.pyMuIn * info.pyMuMinus + info.pzMuIn * info.pzMuMinus) / pProdRadMinus;
            info.openingAngleRadMinusMrad = std::acos(std::clamp(cosTh, -1.0, 1.0)) * 1000.0;
        }
        double pProdRadPlus = info.pMuIn * info.pMuPlus;
        if (pProdRadPlus > 0.0) {
            double cosTh = (info.pxMuIn * info.pxMuPlus + info.pyMuIn * info.pyMuPlus + info.pzMuIn * info.pzMuPlus) / pProdRadPlus;
            info.openingAngleRadPlusMrad = std::acos(std::clamp(cosTh, -1.0, 1.0)) * 1000.0;
        }

        if (eRad > 0.0) {
            info.energyRatioPair = eTot / eRad;
        }
        if (info.pMuIn > 0.0) {
            info.momentumFractionMinus = info.pMuMinus / info.pMuIn;
            info.momentumFractionPlus  = info.pMuPlus / info.pMuIn;
        }
    }

    double ptSum = info.ptMuPlus + info.ptMuMinus;
    info.ptAsymmetry = (ptSum > 0.0) ? (info.ptMuPlus - info.ptMuMinus) / ptSum : 0.0;
}

TridentTruthInfo TridentTruthProcessor::makeEmptyInfo(const TClonesArray& mcTracks) const
{
    TridentTruthInfo empty;
    empty.hasCandidate = false;
    empty.isFiducial = false;
    empty.procType = kUnknown;
    empty.regionType = kRegionUnknown;
    empty.tridentFlags = 0;
    empty.mcWeight = (mcTracks.GetEntriesFast() > 0) ? static_cast<ShipMCTrack*>(mcTracks.At(0))->GetWeight() : 0.0;
    empty.scaledWeight = 0.0;
    return empty;
}

TridentTruthInfo TridentTruthProcessor::process(const TClonesArray& mcTracks) const {
    // 1. Identify opposite-sign muon pair within Z acceptance
    auto candidate = findCandidatePair(mcTracks);
    if (!candidate.has_value()) {
        return makeEmptyInfo(mcTracks);
    }
    const auto [iMinus, iPlus] = *candidate;
    auto* mcTrkMinus = static_cast<ShipMCTrack*>(mcTracks.At(iMinus));
    auto* mcTrkPlus  = static_cast<ShipMCTrack*>(mcTracks.At(iPlus));

    // 2. Trace shower ancestry and classify process
    AncestryResult ancestry = traceAncestry(mcTracks, mcTrkMinus->GetMotherId(), mcTrkMinus->GetProcID());
    if (ancestry.procType >= 0 && ((1 << ancestry.procType) & fConfig.processMask) == 0) {
        return makeEmptyInfo(mcTracks);
    }

    // 3. Determine spatial region
    double zVertex = mcTrkMinus->GetStartZ();
    RegionType region = determineRegion(zVertex);

    // 4. Fiducial containment
    const auto* radMcTrk = getRadiatingTrack(mcTracks, ancestry.radMuonId);
    FiducialResult fid = checkFiducialTriple(mcTrkMinus, mcTrkPlus, radMcTrk);

    // 5. Assemble structured result
    TridentTruthInfo info;
    info.hasCandidate = true;
    info.isFiducial = fid.isFiducial;
    info.procType = ancestry.procType;
    info.regionType = region;
    info.tridentFlags = packFlags(ancestry, region, fid.isFiducial);
    info.mcWeight = mcTrkMinus->GetWeight();
    info.scaledWeight = calculateWeight(info.mcWeight);

    info.vtxX = mcTrkMinus->GetStartX();
    info.vtxY = mcTrkMinus->GetStartY();
    info.vtxZ = zVertex;
    info.radMuonId = ancestry.radMuonId;
    info.muMinusId = iMinus;
    info.muPlusId  = iPlus;
    info.cascadeDepth = ancestry.cascadeDepth;
    info.emissionProc = ancestry.emissionProc;

    info.fidXMinus = fid.xMinus;
    info.fidYMinus = fid.yMinus;
    info.fidXPlus  = fid.xPlus;
    info.fidYPlus  = fid.yPlus;
    info.fidXRad   = fid.xRad;
    info.fidYRad   = fid.yRad;
    info.distToFiducialEdge = fid.distToEdge;

    fillKinematics(info, mcTrkMinus, mcTrkPlus, radMcTrk);

    return info;
}

} // namespace snd::trident

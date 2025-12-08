"""
Unified Frequency Intelligence System

Orchestrates all frequency analyzers for cross-domain threat correlation.
Provides integrated threat detection and actionable countermeasure recommendations.

Features:
- Cross-domain threat correlation
- Probabilistic threat scoring
- Actionable countermeasure recommendations
- Adaptive analysis based on battery level
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

# Import all frequency analyzers
from plugins.rf_spectrum_analyzer import RFSpectrumAnalyzer, RFThreat
from plugins.acoustic_frequency import AcousticFrequencyAnalyzer, AcousticEvent
from plugins.vibration_frequency import VibrationFrequencyAnalyzer, VibrationEvent
from core.network_frequency_analyzer import NetworkFrequencyAnalyzer, NetworkThreat
from core.behavioral_frequency import BehavioralFrequencyAnalyzer, BehaviorThreat
from core.keystroke_frequency import KeystrokeFrequencyAnalyzer, AuthenticationResult

logger = logging.getLogger(__name__)


class ThreatCorrelationType(str, Enum):
    """Types of cross-domain threat correlations."""
    PHYSICAL_CYBER_ATTACK = "physical_cyber_attack"  # RF jamming + network disruption
    APT_CAMPAIGN = "apt_campaign"  # Ultrasonic + C2 beacon
    INSIDER_THREAT = "insider_threat"  # Keystroke anomaly + data exfiltration
    COORDINATED_ATTACK = "coordinated_attack"  # Multiple domains simultaneously
    RECONNAISSANCE = "reconnaissance"  # Port scan + acoustic/vibration monitoring
    NONE = "none"


class IntegratedThreatLevel(str, Enum):
    """Integrated threat severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CorrelatedThreat:
    """Represents a correlated threat across multiple domains."""
    correlation_type: ThreatCorrelationType
    threat_level: IntegratedThreatLevel
    confidence: float
    timestamp: float
    description: str
    domains_involved: List[str]
    contributing_threats: List[Any] = field(default_factory=list)
    countermeasures: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrequencyEnvironment:
    """Snapshot of frequency environment across all domains."""
    timestamp: float
    rf_signals: int
    rf_threats: int
    acoustic_events: int
    vibration_events: int
    network_threats: int
    behavior_threats: int
    authentication_status: str
    overall_threat_score: float
    active_correlations: List[ThreatCorrelationType]


class UnifiedFrequencyIntelligence:
    """
    Unified Frequency Intelligence System.
    
    Orchestrates all frequency analyzers and performs cross-domain correlation
    to detect sophisticated threats that span multiple domains.
    
    Attributes:
        rf_analyzer: RF spectrum analyzer
        acoustic_analyzer: Acoustic frequency analyzer
        vibration_analyzer: Vibration frequency analyzer
        network_analyzer: Network frequency analyzer
        behavior_analyzer: Behavioral frequency analyzer
        keystroke_analyzer: Keystroke frequency analyzer
    """
    
    # Domain weighting factors for threat scoring (configurable)
    WEIGHT_RF = 0.2
    WEIGHT_ACOUSTIC = 0.1
    WEIGHT_VIBRATION = 0.05
    WEIGHT_NETWORK = 0.25
    WEIGHT_BEHAVIORAL = 0.3
    WEIGHT_AUTH_REJECTED = 0.5
    WEIGHT_AUTH_SUSPICIOUS = 0.2
    
    def __init__(
        self,
        enable_rf: bool = True,
        enable_acoustic: bool = True,
        enable_vibration: bool = True,
        enable_network: bool = True,
        enable_behavioral: bool = True,
        enable_keystroke: bool = True
    ):
        """
        Initialize Unified Frequency Intelligence System.
        
        Args:
            enable_rf: Enable RF spectrum analysis
            enable_acoustic: Enable acoustic analysis
            enable_vibration: Enable vibration analysis
            enable_network: Enable network analysis
            enable_behavioral: Enable behavioral analysis
            enable_keystroke: Enable keystroke analysis
        """
        # Initialize analyzers
        self.rf_analyzer = RFSpectrumAnalyzer() if enable_rf else None
        self.acoustic_analyzer = AcousticFrequencyAnalyzer() if enable_acoustic else None
        self.vibration_analyzer = VibrationFrequencyAnalyzer() if enable_vibration else None
        self.network_analyzer = NetworkFrequencyAnalyzer() if enable_network else None
        self.behavior_analyzer = BehavioralFrequencyAnalyzer() if enable_behavioral else None
        self.keystroke_analyzer = KeystrokeFrequencyAnalyzer() if enable_keystroke else None
        
        # Correlated threats
        self.correlated_threats: deque = deque(maxlen=100)
        
        # Environment history
        self.environment_history: deque = deque(maxlen=50)
        
        # Baseline environment
        self.baseline_environment: Optional[FrequencyEnvironment] = None
        
        logger.info("Unified Frequency Intelligence System initialized")
    
    def analyze_all_frequencies(
        self,
        battery_level: float = 1.0,
        duration_seconds: float = 1.0
    ) -> FrequencyEnvironment:
        """
        Analyze all frequency domains.
        
        Args:
            battery_level: Current battery level (0.0 to 1.0) for adaptive analysis
            duration_seconds: Duration to analyze
            
        Returns:
            Frequency environment snapshot
        """
        timestamp = time.time()
        
        # Adaptive analysis based on battery
        adaptive_duration = duration_seconds if battery_level > 0.3 else duration_seconds * 0.5
        
        # RF analysis
        rf_signals = 0
        rf_threats = 0
        if self.rf_analyzer:
            try:
                signals = self.rf_analyzer.scan_spectrum(adaptive_duration, battery_level=battery_level)
                rf_signals = len(signals)
                
                jamming = self.rf_analyzer.detect_jamming()
                rogues = self.rf_analyzer.detect_rogue_transmitters()
                rf_threats = (1 if jamming else 0) + len(rogues)
            except Exception as e:
                logger.error(f"RF analysis error: {e}")
        
        # Acoustic analysis
        acoustic_events = 0
        if self.acoustic_analyzer:
            try:
                events = self.acoustic_analyzer.analyze_audio_sample(duration_seconds=adaptive_duration)
                acoustic_events = len(events)
            except Exception as e:
                logger.error(f"Acoustic analysis error: {e}")
        
        # Vibration analysis
        vibration_events = 0
        if self.vibration_analyzer:
            try:
                events = self.vibration_analyzer.analyze_vibration(duration_seconds=adaptive_duration)
                vibration_events = len(events)
            except Exception as e:
                logger.error(f"Vibration analysis error: {e}")
        
        # Network analysis
        network_threats = 0
        if self.network_analyzer:
            try:
                results = self.network_analyzer.analyze_all()
                network_threats = sum(len(threats) for threats in results.values())
            except Exception as e:
                logger.error(f"Network analysis error: {e}")
        
        # Behavioral analysis
        behavior_threats = 0
        if self.behavior_analyzer:
            try:
                results = self.behavior_analyzer.analyze_all_processes()
                behavior_threats = sum(len(threats) for threats in results.values())
            except Exception as e:
                logger.error(f"Behavioral analysis error: {e}")
        
        # Keystroke authentication
        auth_status = "unknown"
        if self.keystroke_analyzer:
            try:
                # Check recent authentications
                if self.keystroke_analyzer.authentication_results:
                    recent = self.keystroke_analyzer.authentication_results[-1]
                    auth_status = recent.status.value
            except Exception as e:
                logger.error(f"Keystroke analysis error: {e}")
        
        # Calculate overall threat score
        threat_score = self._calculate_threat_score(
            rf_threats, acoustic_events, vibration_events,
            network_threats, behavior_threats, auth_status
        )
        
        # Correlate threats
        active_correlations = self._correlate_threats()
        
        # Create environment snapshot
        environment = FrequencyEnvironment(
            timestamp=timestamp,
            rf_signals=rf_signals,
            rf_threats=rf_threats,
            acoustic_events=acoustic_events,
            vibration_events=vibration_events,
            network_threats=network_threats,
            behavior_threats=behavior_threats,
            authentication_status=auth_status,
            overall_threat_score=threat_score,
            active_correlations=active_correlations
        )
        
        self.environment_history.append(environment)
        
        # Establish baseline if not set
        if self.baseline_environment is None and len(self.environment_history) >= 10:
            self._establish_baseline()
        
        return environment
    
    def _calculate_threat_score(
        self,
        rf_threats: int,
        acoustic_events: int,
        vibration_events: int,
        network_threats: int,
        behavior_threats: int,
        auth_status: str
    ) -> float:
        """
        Calculate overall threat score (0.0 to 1.0).
        
        Uses configurable domain weighting factors to balance contributions
        from different threat detection domains.
        
        Args:
            rf_threats: Number of RF threats
            acoustic_events: Number of acoustic events
            vibration_events: Number of vibration events
            network_threats: Number of network threats
            behavior_threats: Number of behavioral threats
            auth_status: Authentication status
            
        Returns:
            Overall threat score
        """
        score = 0.0
        
        # Weight each domain using class constants
        score += min(1.0, rf_threats * self.WEIGHT_RF)
        score += min(1.0, acoustic_events * self.WEIGHT_ACOUSTIC)
        score += min(1.0, vibration_events * self.WEIGHT_VIBRATION)
        score += min(1.0, network_threats * self.WEIGHT_NETWORK)
        score += min(1.0, behavior_threats * self.WEIGHT_BEHAVIORAL)
        
        # Authentication failures are critical
        if auth_status == "rejected":
            score += self.WEIGHT_AUTH_REJECTED
        elif auth_status == "suspicious":
            score += self.WEIGHT_AUTH_SUSPICIOUS
        
        return min(1.0, score)
    
    def _correlate_threats(self) -> List[ThreatCorrelationType]:
        """
        Correlate threats across domains to identify sophisticated attacks.
        
        Returns:
            List of active threat correlations
        """
        correlations = []
        
        # Gather recent threats
        rf_threats = self.rf_analyzer.detected_threats[-10:] if self.rf_analyzer else []
        acoustic_events = self.acoustic_analyzer.detected_events[-10:] if self.acoustic_analyzer else []
        network_threats = self.network_analyzer.detected_threats[-10:] if self.network_analyzer else []
        behavior_threats = self.behavior_analyzer.detected_threats[-10:] if self.behavior_analyzer else []
        auth_results = self.keystroke_analyzer.authentication_results[-10:] if self.keystroke_analyzer else []
        
        # Check for Physical + Cyber Attack
        # RF jamming + Network disruption
        has_rf_jamming = any(t.threat_type == "jamming" for t in rf_threats)
        has_network_disruption = any(t.threat_level.value in ["high", "critical"] for t in network_threats)
        
        if has_rf_jamming and has_network_disruption:
            correlation = self._create_correlation(
                ThreatCorrelationType.PHYSICAL_CYBER_ATTACK,
                IntegratedThreatLevel.CRITICAL,
                "RF jamming combined with network disruption detected",
                ["rf", "network"],
                rf_threats + network_threats
            )
            self.correlated_threats.append(correlation)
            correlations.append(ThreatCorrelationType.PHYSICAL_CYBER_ATTACK)
        
        # Check for APT Campaign
        # Ultrasonic communication + C2 beacon
        has_ultrasonic = any(e.signature.value == "ultrasonic_communication" for e in acoustic_events)
        has_c2_beacon = any(t.threat_type.value == "c2_beacon" for t in network_threats)
        
        if has_ultrasonic and has_c2_beacon:
            correlation = self._create_correlation(
                ThreatCorrelationType.APT_CAMPAIGN,
                IntegratedThreatLevel.CRITICAL,
                "Covert ultrasonic communication with C2 beaconing detected - possible APT",
                ["acoustic", "network"],
                acoustic_events + network_threats
            )
            self.correlated_threats.append(correlation)
            correlations.append(ThreatCorrelationType.APT_CAMPAIGN)
        
        # Check for Insider Threat
        # Keystroke anomaly + Behavioral threat
        has_auth_failure = any(r.status.value in ["rejected", "suspicious"] for r in auth_results)
        has_malicious_behavior = any(t.severity.value in ["malicious", "critical"] for t in behavior_threats)
        
        if has_auth_failure and has_malicious_behavior:
            correlation = self._create_correlation(
                ThreatCorrelationType.INSIDER_THREAT,
                IntegratedThreatLevel.HIGH,
                "Authentication anomaly with malicious behavior - possible insider threat",
                ["keystroke", "behavioral"],
                auth_results + behavior_threats
            )
            self.correlated_threats.append(correlation)
            correlations.append(ThreatCorrelationType.INSIDER_THREAT)
        
        # Check for Reconnaissance
        # Port scan + Acoustic/Vibration monitoring
        has_port_scan = any(t.threat_type.value == "port_scan" for t in network_threats)
        has_acoustic_signatures = len(acoustic_events) > 0
        
        if has_port_scan and has_acoustic_signatures:
            correlation = self._create_correlation(
                ThreatCorrelationType.RECONNAISSANCE,
                IntegratedThreatLevel.MEDIUM,
                "Port scanning with acoustic monitoring - possible reconnaissance",
                ["network", "acoustic"],
                network_threats + acoustic_events
            )
            self.correlated_threats.append(correlation)
            correlations.append(ThreatCorrelationType.RECONNAISSANCE)
        
        # Check for Coordinated Attack
        # Multiple domains with threats simultaneously
        active_domains = 0
        if rf_threats:
            active_domains += 1
        if acoustic_events:
            active_domains += 1
        if network_threats:
            active_domains += 1
        if behavior_threats:
            active_domains += 1
        
        if active_domains >= 3:
            all_threats = rf_threats + acoustic_events + network_threats + behavior_threats
            correlation = self._create_correlation(
                ThreatCorrelationType.COORDINATED_ATTACK,
                IntegratedThreatLevel.CRITICAL,
                f"Coordinated attack across {active_domains} domains detected",
                ["rf", "acoustic", "network", "behavioral"][:active_domains],
                all_threats
            )
            self.correlated_threats.append(correlation)
            correlations.append(ThreatCorrelationType.COORDINATED_ATTACK)
        
        return correlations
    
    def _create_correlation(
        self,
        correlation_type: ThreatCorrelationType,
        threat_level: IntegratedThreatLevel,
        description: str,
        domains: List[str],
        threats: List[Any]
    ) -> CorrelatedThreat:
        """
        Create a correlated threat with countermeasures.
        
        Args:
            correlation_type: Type of correlation
            threat_level: Threat severity
            description: Description
            domains: Involved domains
            threats: Contributing threats
            
        Returns:
            Correlated threat object
        """
        # Determine countermeasures based on correlation type
        countermeasures = self._generate_countermeasures(correlation_type, domains)
        
        # Calculate confidence
        confidence = min(1.0, len(threats) / 5.0)
        
        correlation = CorrelatedThreat(
            correlation_type=correlation_type,
            threat_level=threat_level,
            confidence=confidence,
            timestamp=time.time(),
            description=description,
            domains_involved=domains,
            contributing_threats=threats,
            countermeasures=countermeasures,
            metadata={"threat_count": len(threats)}
        )
        
        logger.warning(f"Correlated threat: {correlation_type.value} - {description}")
        
        return correlation
    
    def _generate_countermeasures(
        self,
        correlation_type: ThreatCorrelationType,
        domains: List[str]
    ) -> List[str]:
        """
        Generate actionable countermeasures for a threat correlation.
        
        Args:
            correlation_type: Type of threat correlation
            domains: Involved domains
            
        Returns:
            List of countermeasure recommendations
        """
        countermeasures = []
        
        if correlation_type == ThreatCorrelationType.PHYSICAL_CYBER_ATTACK:
            countermeasures = [
                "Switch to backup RF channels",
                "Enable network redundancy protocols",
                "Activate physical security measures",
                "Isolate critical systems",
                "Alert security team for immediate response"
            ]
        
        elif correlation_type == ThreatCorrelationType.APT_CAMPAIGN:
            countermeasures = [
                "Block ultrasonic frequencies (>18kHz)",
                "Isolate systems with C2 beaconing",
                "Enable advanced packet inspection",
                "Conduct full malware scan",
                "Escalate to incident response team"
            ]
        
        elif correlation_type == ThreatCorrelationType.INSIDER_THREAT:
            countermeasures = [
                "Lock user account immediately",
                "Require multi-factor re-authentication",
                "Review recent file access logs",
                "Monitor data exfiltration attempts",
                "Alert security operations center"
            ]
        
        elif correlation_type == ThreatCorrelationType.RECONNAISSANCE:
            countermeasures = [
                "Block scanning IP addresses",
                "Enable honeypot systems",
                "Increase logging and monitoring",
                "Review acoustic sensor data for patterns",
                "Prepare for potential follow-up attack"
            ]
        
        elif correlation_type == ThreatCorrelationType.COORDINATED_ATTACK:
            countermeasures = [
                "Activate emergency defense protocols",
                "Isolate all critical systems",
                "Enable all redundancy measures",
                "Alert all security teams",
                "Prepare for incident response and recovery"
            ]
        
        return countermeasures
    
    def _establish_baseline(self):
        """Establish baseline environment from history."""
        if len(self.environment_history) < 10:
            return
        
        # Calculate average environment
        recent = list(self.environment_history)[-10:]
        
        avg_env = FrequencyEnvironment(
            timestamp=time.time(),
            rf_signals=int(np.mean([e.rf_signals for e in recent])),
            rf_threats=int(np.mean([e.rf_threats for e in recent])),
            acoustic_events=int(np.mean([e.acoustic_events for e in recent])),
            vibration_events=int(np.mean([e.vibration_events for e in recent])),
            network_threats=int(np.mean([e.network_threats for e in recent])),
            behavior_threats=int(np.mean([e.behavior_threats for e in recent])),
            authentication_status="authenticated",
            overall_threat_score=float(np.mean([e.overall_threat_score for e in recent])),
            active_correlations=[]
        )
        
        self.baseline_environment = avg_env
        logger.info("Baseline frequency environment established")
    
    def get_threat_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive threat report.
        
        Returns:
            Dictionary with threat intelligence
        """
        current_env = self.environment_history[-1] if self.environment_history else None
        
        report = {
            "timestamp": time.time(),
            "current_environment": {
                "overall_threat_score": current_env.overall_threat_score if current_env else 0.0,
                "rf_signals": current_env.rf_signals if current_env else 0,
                "rf_threats": current_env.rf_threats if current_env else 0,
                "acoustic_events": current_env.acoustic_events if current_env else 0,
                "vibration_events": current_env.vibration_events if current_env else 0,
                "network_threats": current_env.network_threats if current_env else 0,
                "behavior_threats": current_env.behavior_threats if current_env else 0,
                "authentication_status": current_env.authentication_status if current_env else "unknown"
            },
            "active_correlations": [c.value for c in current_env.active_correlations] if current_env else [],
            "recent_correlated_threats": len(self.correlated_threats),
            "analyzer_status": {
                "rf_enabled": self.rf_analyzer is not None,
                "acoustic_enabled": self.acoustic_analyzer is not None,
                "vibration_enabled": self.vibration_analyzer is not None,
                "network_enabled": self.network_analyzer is not None,
                "behavioral_enabled": self.behavior_analyzer is not None,
                "keystroke_enabled": self.keystroke_analyzer is not None
            },
            "recommendations": self._get_recommendations()
        }
        
        return report
    
    def _get_recommendations(self) -> List[str]:
        """Generate recommendations based on current threat landscape."""
        recommendations = []
        
        if not self.correlated_threats:
            recommendations.append("Continue normal monitoring")
            return recommendations
        
        # Get recent critical correlations
        recent_critical = [
            c for c in self.correlated_threats
            if c.threat_level == IntegratedThreatLevel.CRITICAL
        ]
        
        if recent_critical:
            recommendations.append("CRITICAL: Immediate action required")
            for threat in recent_critical[-3:]:  # Last 3 critical threats
                recommendations.extend(threat.countermeasures[:2])  # Top 2 countermeasures
        else:
            recommendations.append("Maintain elevated security posture")
            recommendations.append("Continue monitoring all frequency domains")
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get unified intelligence statistics."""
        stats = {
            "environment_snapshots": len(self.environment_history),
            "correlated_threats": len(self.correlated_threats),
            "baseline_established": self.baseline_environment is not None
        }
        
        # Add per-analyzer stats
        if self.rf_analyzer:
            stats["rf_analyzer"] = self.rf_analyzer.get_statistics()
        if self.acoustic_analyzer:
            stats["acoustic_analyzer"] = self.acoustic_analyzer.get_statistics()
        if self.vibration_analyzer:
            stats["vibration_analyzer"] = self.vibration_analyzer.get_statistics()
        if self.network_analyzer:
            stats["network_analyzer"] = self.network_analyzer.get_statistics()
        if self.behavior_analyzer:
            stats["behavior_analyzer"] = self.behavior_analyzer.get_statistics()
        if self.keystroke_analyzer:
            stats["keystroke_analyzer"] = self.keystroke_analyzer.get_statistics()
        
        return stats

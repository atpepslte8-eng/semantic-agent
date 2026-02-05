"""
Security Module - Nexus Fortress Integration for Semantic Agent

Protects against:
- Memory poisoning attacks
- Prompt injection via knowledge base
- Unbounded/dangerous actions
- Untrusted source contamination
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SecurityGate:
    """Security checkpoint for actions and knowledge ingestion"""
    
    def __init__(self):
        self.injection_patterns = [
            # Direct injection attempts
            r"ignore\s+previous\s+instructions",
            r"forget\s+your\s+rules",
            r"system\s+override:",
            r"admin\s+mode:",
            r"debug\s+mode:",
            r"new\s+instructions:",
            r"actually,?\s+disregard",
            r"you\s+are\s+now\s+(?:unrestricted|DAN|jailbroken)",
            
            # Hidden injection vectors
            r"<!--.*?execute.*?-->",
            r"<script.*?</script>",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            
            # Social engineering
            r"the\s+user\s+authorized\s+this",
            r"i'm\s+the\s+admin",
            r"this\s+is\s+an\s+emergency",
            r"bypass\s+safety",
            r"for\s+testing\s+purposes",
        ]
        
        self.dangerous_actions = {
            'file_delete', 'system_command', 'network_request',
            'credential_access', 'admin_action', 'external_send'
        }
        
        # Trusted sources (owner identification)
        self.trusted_numbers = ['+13343223979']  # Innovator's WhatsApp
        self.unlock_commands = ['!unlock', '!dev off-all', '!reset security']
        
    def scan_content(self, content: str, source: str = "unknown") -> Tuple[ThreatLevel, List[str]]:
        """Scan content for injection attempts"""
        threats = []
        max_level = ThreatLevel.NONE
        
        # Skip scanning if from trusted source (owner)
        if self._is_trusted_source(source):
            return ThreatLevel.NONE, []
        
        content_lower = content.lower()
        
        # Check for injection patterns
        for pattern in self.injection_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE | re.MULTILINE):
                threats.append(f"Injection pattern detected: {pattern}")
                max_level = max(max_level, ThreatLevel.HIGH)
        
        # Check for encoded/hidden content
        if self._check_encoded_threats(content):
            threats.append("Suspicious encoded content detected")
            max_level = max(max_level, ThreatLevel.MEDIUM)
        
        # Check for the "Lethal Trifecta"
        trifecta_score = self._check_lethal_trifecta(content, source)
        if trifecta_score >= 2:
            threats.append(f"Lethal Trifecta: {trifecta_score}/3 indicators")
            max_level = max(max_level, ThreatLevel.CRITICAL)
        
        return max_level, threats
    
    def validate_action(self, action_type: str, action_data: Dict[str, Any], 
                       source: str = "unknown") -> Tuple[bool, str]:
        """Validate if an action should be allowed"""
        
        # Owner can do anything
        if self._is_trusted_source(source):
            return True, "Trusted source - allowing all actions"
        
        # Check for emergency unlock commands
        if isinstance(action_data.get('command'), str):
            for unlock_cmd in self.unlock_commands:
                if unlock_cmd in action_data['command'].lower():
                    return True, f"Emergency unlock command: {unlock_cmd}"
        
        # Block dangerous actions from untrusted sources
        if action_type in self.dangerous_actions:
            return False, f"Dangerous action '{action_type}' blocked from untrusted source"
        
        return True, "Action allowed"
    
    def sanitize_memory_input(self, knowledge: str, source: str = "unknown") -> str:
        """Sanitize knowledge before storing in memory"""
        
        # Don't sanitize owner input
        if self._is_trusted_source(source):
            return knowledge
        
        # Remove potential injection attempts
        sanitized = knowledge
        for pattern in self.injection_patterns:
            sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)
        
        # Add source metadata
        sanitized = f"[SOURCE: {source}]\n{sanitized}"
        
        return sanitized
    
    def _is_trusted_source(self, source: str) -> bool:
        """Check if source is the trusted owner"""
        # Check for owner phone number
        for trusted in self.trusted_numbers:
            if trusted in source:
                return True
        
        # Check for main session (direct chat)
        if source in ['main_session', 'direct_chat', 'owner']:
            return True
            
        return False
    
    def _check_encoded_threats(self, content: str) -> bool:
        """Check for base64 or other encoded threats"""
        import base64
        
        # Look for base64 patterns
        b64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        matches = re.findall(b64_pattern, content)
        
        for match in matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                # Check if decoded content contains suspicious patterns
                if any(re.search(pattern, decoded, re.IGNORECASE) 
                       for pattern in self.injection_patterns[:5]):  # Check first 5 patterns
                    return True
            except:
                continue
                
        return False
    
    def _check_lethal_trifecta(self, content: str, source: str) -> int:
        """Check for the three danger indicators"""
        score = 0
        
        # 1. Involves private/sensitive data?
        sensitive_keywords = ['password', 'token', 'key', 'credential', 'private', 'secret']
        if any(keyword in content.lower() for keyword in sensitive_keywords):
            score += 1
        
        # 2. Instruction from external/untrusted source?
        if not self._is_trusted_source(source):
            score += 1
        
        # 3. Would send data externally or execute code?
        external_keywords = ['send', 'post', 'upload', 'execute', 'run', 'eval', 'exec']
        if any(keyword in content.lower() for keyword in external_keywords):
            score += 1
        
        return score

class SecurityLogger:
    """Logs security events for analysis"""
    
    def __init__(self, log_file: str = "~/.openclaw/security-sentinel.log"):
        self.log_file = log_file
        
    def log_threat(self, threat_level: ThreatLevel, threats: List[str], 
                  content_hash: str, source: str):
        """Log detected threats"""
        event = {
            'timestamp': self._get_timestamp(),
            'threat_level': threat_level.name,
            'threats': threats,
            'content_hash': content_hash,
            'source': source
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    def _get_timestamp(self) -> str:
        import datetime
        return datetime.datetime.now().isoformat()

class SecureAgent:
    """Security-aware wrapper for agent operations"""
    
    def __init__(self, agent):
        self.agent = agent
        self.security_gate = SecurityGate()
        self.security_logger = SecurityLogger()
    
    def secure_think(self, prompt: str, source: str = "unknown") -> str:
        """Secure wrapper for agent thinking"""
        
        # Scan input for threats
        threat_level, threats = self.security_gate.scan_content(prompt, source)
        
        if threat_level >= ThreatLevel.HIGH:
            # Log the threat
            content_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            self.security_logger.log_threat(threat_level, threats, content_hash, source)
            
            # Return security alert instead of processing
            return self._generate_security_alert(threat_level, threats, source)
        
        # Process normally if safe
        return self.agent.think(prompt)
    
    def secure_action(self, action_type: str, action_data: Dict[str, Any], 
                     source: str = "unknown") -> Tuple[bool, str]:
        """Secure wrapper for agent actions"""
        
        allowed, reason = self.security_gate.validate_action(action_type, action_data, source)
        
        if not allowed:
            # Log blocked action
            self.security_logger.log_threat(
                ThreatLevel.HIGH, 
                [f"Blocked action: {action_type}"],
                hashlib.sha256(str(action_data).encode()).hexdigest()[:16],
                source
            )
        
        return allowed, reason
    
    def secure_memory_store(self, knowledge: str, source: str = "unknown") -> str:
        """Secure wrapper for memory storage"""
        
        # Scan and sanitize
        threat_level, threats = self.security_gate.scan_content(knowledge, source)
        
        if threat_level >= ThreatLevel.CRITICAL:
            # Refuse to store highly dangerous content
            return None
        
        # Sanitize before storing
        sanitized = self.security_gate.sanitize_memory_input(knowledge, source)
        
        return sanitized
    
    def _generate_security_alert(self, threat_level: ThreatLevel, threats: List[str], 
                                source: str) -> str:
        """Generate security alert message"""
        
        alert = f"""🚨 SECURITY ALERT

I detected a prompt injection attempt.
Threat Level: {threat_level.name}
Source: {source}
Patterns: {', '.join(threats)}

I have REFUSED to execute these instructions.
The content is being treated as untrusted data only.
"""
        return alert

# Utility functions for easy integration
def create_secure_agent(agent):
    """Create a security-wrapped agent"""
    return SecureAgent(agent)

def scan_for_threats(content: str, source: str = "unknown") -> Tuple[ThreatLevel, List[str]]:
    """Standalone threat scanning"""
    gate = SecurityGate()
    return gate.scan_content(content, source)
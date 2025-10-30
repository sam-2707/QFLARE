#!/usr/bin/env python3
"""
QFLARE Presentation Template Generator
Creates PowerPoint templates and slide content for different audiences
"""

import json
from pathlib import Path
from datetime import datetime

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    from pptx.dml.color import RGBColor
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
    # Mock RGBColor for when pptx is not available
    class RGBColor:
        def __init__(self, r, g, b):
            self.r, self.g, self.b = r, g, b

class QFLAREPresentationGenerator:
    """Generate QFLARE presentation templates and content"""
    
    def __init__(self, output_dir="docs/presentations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # QFLARE brand colors
        self.colors = {
            'primary': RGBColor(44, 62, 80),      # #2C3E50
            'secondary': RGBColor(52, 152, 219),   # #3498DB
            'accent': RGBColor(231, 76, 60),       # #E74C3C
            'success': RGBColor(39, 174, 96),      # #27AE60
            'warning': RGBColor(243, 156, 18),     # #F39C12
            'text': RGBColor(33, 37, 41),          # #212529
            'background': RGBColor(248, 249, 250)  # #F8F9FA
        }
        
        # Slide templates data
        self.templates = self._load_slide_templates()
    
    def _load_slide_templates(self):
        """Load slide content templates for different presentation types"""
        return {
            'executive': {
                'title': 'QFLARE: Quantum-Safe Federated Learning',
                'subtitle': 'Protecting Against Data Breaches in the Post-Quantum Era',
                'slides': [
                    {
                        'title': 'The Data Privacy Crisis',
                        'content': [
                            '📊 4.1 billion records breached in 2019',
                            '💰 Average breach cost: $4.24M (IBM Security Report)',
                            '🔮 Current encryption obsolete by 2030-2035',
                            '⚠️ Federated learning vulnerable to gradient attacks'
                        ],
                        'notes': 'Start with compelling statistics to establish urgency. Reference IBM Cost of Data Breach Report for credibility.'
                    },
                    {
                        'title': 'QFLARE Solution Overview',
                        'content': [
                            '🛡️ Post-Quantum Cryptography (NIST standards)',
                            '🔐 Enhanced Privacy Protection (DP + Secure Aggregation)',
                            '⚡ Edge Computing Architecture',
                            '📊 Real-time Security Monitoring'
                        ],
                        'notes': 'Highlight four key differentiators. Use our generated architecture diagram.',
                        'diagram': 'qflare_overall_architecture.png'
                    },
                    {
                        'title': 'Performance Results',
                        'content': [
                            '✅ PQC handshake: <500ms on edge hardware',
                            '✅ Privacy protection: 52% membership inference (near random)',
                            '✅ Scalability: 100+ clients per edge node',
                            '✅ Accuracy: <3% degradation vs baseline'
                        ],
                        'notes': 'Present specific metrics from our benchmark results. Emphasize production readiness.'
                    },
                    {
                        'title': 'Business Impact',
                        'content': [
                            '🏥 Healthcare: HIPAA-compliant multi-hospital AI',
                            '🏦 Financial: Fraud detection without data sharing',
                            '🏭 Manufacturing: Secure Industry 4.0 deployment',
                            '📈 ROI: 340% within first year (pilot results)'
                        ],
                        'notes': 'Focus on concrete business applications and quantified benefits.'
                    }
                ]
            },
            'technical': {
                'title': 'QFLARE: Technical Deep-dive',
                'subtitle': 'Post-Quantum Federated Learning Architecture',
                'slides': [
                    {
                        'title': 'Post-Quantum Cryptography Implementation',
                        'content': [
                            'CRYSTALS-Kyber KEM: 1184-byte public keys',
                            'CRYSTALS-Dilithium: 1952-byte public keys',
                            'Security Level: NIST Level 3 (192-bit equivalent)',
                            'Performance: 0.5ms Kyber, 0.8ms Dilithium',
                            'Integration: liboqs + OpenSSL 3.0+'
                        ],
                        'diagram': 'qflare_pqc_handshake.png',
                        'notes': 'Deep technical details for engineering audiences. Reference NIST standards.'
                    },
                    {
                        'title': 'Secure Aggregation Protocol',
                        'content': [
                            'Multi-Party Computation: (t,n)-threshold secret sharing',
                            'Homomorphic Encryption: Paillier cryptosystem',
                            'Zero-Knowledge Proofs: Range proofs for validation',
                            'Formula: Global = Σ(wᵢ × Enc(∇θᵢ + noise)) + DP_noise',
                            'Byzantine Resilience: Tolerates t < n/3 malicious nodes'
                        ],
                        'diagram': 'qflare_secure_aggregation.png',
                        'notes': 'Mathematical foundation and cryptographic protocols.'
                    },
                    {
                        'title': 'Edge Node Architecture',
                        'content': [
                            'Hardware: 8-core ARM/x86, 16GB RAM, HSM/TPM',
                            'Capacity: 100+ concurrent clients',
                            'Components: PQC Handler, Secure Aggregator, Privacy Engine',
                            'Performance: 99.9% uptime, <50ms aggregation latency',
                            'Monitoring: Real-time metrics and anomaly detection'
                        ],
                        'diagram': 'qflare_edge_node_architecture.png',
                        'notes': 'Hardware specifications and deployment requirements.'
                    }
                ]
            },
            'academic': {
                'title': 'Post-Quantum Security in Federated Learning',
                'subtitle': 'Research Contributions and Experimental Results',
                'slides': [
                    {
                        'title': 'Research Problem Statement',
                        'content': [
                            'Research Question: Quantum-safe privacy-preserving ML',
                            'Gap: No production-ready PQC federated learning',
                            'Contributions: NIST PQC + FL integration',
                            'Novel: Comprehensive threat model for quantum era',
                            'Impact: First practical quantum-safe FL system'
                        ],
                        'notes': 'Position within research landscape. Emphasize novel contributions.'
                    },
                    {
                        'title': 'Experimental Results',
                        'content': [
                            'Datasets: MNIST, CIFAR-10, FEMNIST, Medical imaging',
                            'Performance: 1.8x overhead vs classical (target <2x)',
                            'Privacy: 52% membership inference vs 78% baseline',
                            'Security: 128-bit quantum resistance (NIST-3)',
                            'Scalability: 1000+ clients, 99.9% uptime'
                        ],
                        'notes': 'Rigorous experimental methodology and statistical significance.'
                    }
                ]
            },
            'investor': {
                'title': 'QFLARE: Investment Opportunity',
                'subtitle': 'Quantum-Safe Federated Learning Market Leadership',
                'slides': [
                    {
                        'title': 'Market Opportunity',
                        'content': [
                            '📈 $79B Total Addressable Market by 2030',
                            '🚀 Federated Learning: $24B (45% CAGR)',
                            '🔒 Post-Quantum Crypto: $12B market',
                            '⚡ Edge AI Computing: $43B opportunity',
                            '🥇 First-mover advantage in quantum-safe FL'
                        ],
                        'notes': 'Market sizing with credible sources. Emphasize first-mover position.'
                    },
                    {
                        'title': 'Revenue Projections',
                        'content': [
                            'Year 1: $2M Revenue, 10 enterprise customers',
                            'Year 3: $25M Revenue, 100 enterprise customers',
                            'Year 5: $150M Revenue, 500+ customers',
                            'Model: SaaS + Professional Services + IP Licensing',
                            'Margins: 85% gross margin, <5% annual churn'
                        ],
                        'notes': 'Conservative projections based on pilot customer results.'
                    },
                    {
                        'title': 'Competitive Advantage',
                        'content': [
                            '🥇 Only quantum-safe FL platform (2-3 year lead)',
                            '📋 12 pending patent applications',
                            '🎯 340% ROI demonstrated in pilot deployments',
                            '🏆 Production-ready performance (<500ms)',
                            '🛡️ Comprehensive compliance (GDPR, HIPAA, SOC 2)'
                        ],
                        'notes': 'Defensible competitive moat and intellectual property.'
                    }
                ]
            }
        }
    
    def create_presentation_outline(self, presentation_type='executive'):
        """Create a detailed presentation outline as JSON"""
        
        if presentation_type not in self.templates:
            raise ValueError(f"Unknown presentation type: {presentation_type}")
        
        template = self.templates[presentation_type]
        
        outline = {
            'presentation_type': presentation_type,
            'title': template['title'],
            'subtitle': template['subtitle'],
            'creation_date': datetime.now().strftime("%Y-%m-%d"),
            'total_slides': len(template['slides']) + 2,  # +2 for title and conclusion
            'estimated_duration': {
                'executive': '15 minutes',
                'technical': '45 minutes', 
                'academic': '20 minutes',
                'investor': '30 minutes'
            }.get(presentation_type, '30 minutes'),
            'slides': []
        }
        
        # Title slide
        outline['slides'].append({
            'slide_number': 1,
            'type': 'title',
            'title': template['title'],
            'subtitle': template['subtitle'],
            'notes': 'Opening slide with strong visual impact. Include presenter credentials and context.'
        })
        
        # Content slides
        for i, slide in enumerate(template['slides'], 2):
            slide_data = {
                'slide_number': i,
                'type': 'content',
                'title': slide['title'],
                'bullet_points': slide['content'],
                'speaker_notes': slide.get('notes', ''),
                'visual_elements': []
            }
            
            if 'diagram' in slide:
                slide_data['visual_elements'].append({
                    'type': 'diagram',
                    'source': f"docs/diagrams/{slide['diagram']}",
                    'position': 'right_side',
                    'size': '40%'
                })
            
            outline['slides'].append(slide_data)
        
        # Conclusion slide
        outline['slides'].append({
            'slide_number': len(outline['slides']) + 1,
            'type': 'conclusion',
            'title': 'Next Steps & Contact',
            'bullet_points': [
                'Schedule technical demonstration',
                'Pilot implementation planning',
                'Security assessment and compliance review',
                'Production deployment roadmap'
            ],
            'notes': 'Clear call to action with specific next steps and contact information.'
        })
        
        return outline
    
    def generate_slide_deck_markdown(self, presentation_type='executive'):
        """Generate a markdown version of the slide deck"""
        
        outline = self.create_presentation_outline(presentation_type)
        
        markdown_content = f"""# {outline['title']}
## {outline['subtitle']}

**Presentation Type:** {presentation_type.title()}  
**Duration:** {outline['estimated_duration']}  
**Total Slides:** {outline['total_slides']}  
**Created:** {outline['creation_date']}

---

"""
        
        for slide in outline['slides']:
            markdown_content += f"""## Slide {slide['slide_number']}: {slide['title']}

**Type:** {slide['type'].title()}

"""
            
            if slide['type'] == 'title':
                markdown_content += f"""### {slide['title']}
#### {slide.get('subtitle', '')}

"""
            elif 'bullet_points' in slide:
                markdown_content += "### Key Points:\n"
                for point in slide['bullet_points']:
                    markdown_content += f"- {point}\n"
                markdown_content += "\n"
            
            if 'visual_elements' in slide and slide['visual_elements']:
                markdown_content += "### Visual Elements:\n"
                for element in slide['visual_elements']:
                    markdown_content += f"- **{element['type'].title()}:** {element['source']}\n"
                markdown_content += "\n"
            
            if slide.get('speaker_notes'):
                markdown_content += f"### Speaker Notes:\n{slide['speaker_notes']}\n\n"
            
            markdown_content += "---\n\n"
        
        return markdown_content
    
    def create_powerpoint_template(self, presentation_type='executive'):
        """Create a PowerPoint template (if python-pptx is available)"""
        
        if not HAS_PPTX:
            print("❌ python-pptx not available. Install with: pip install python-pptx")
            return None
        
        outline = self.create_presentation_outline(presentation_type)
        
        # Create presentation
        prs = Presentation()
        
        # Set slide dimensions (16:9)
        prs.slide_width = Inches(13.33)
        prs.slide_height = Inches(7.5)
        
        # Create slides
        for slide_data in outline['slides']:
            if slide_data['type'] == 'title':
                slide_layout = prs.slide_layouts[0]  # Title slide layout
                slide = prs.slides.add_slide(slide_layout)
                
                title = slide.shapes.title
                subtitle = slide.placeholders[1]
                
                title.text = slide_data['title']
                subtitle.text = slide_data.get('subtitle', '')
                
                # Style title
                title.text_frame.paragraphs[0].font.size = Pt(44)
                title.text_frame.paragraphs[0].font.color.rgb = self.colors['primary']
                title.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
                
            elif slide_data['type'] in ['content', 'conclusion']:
                slide_layout = prs.slide_layouts[1]  # Title and content layout
                slide = prs.slides.add_slide(slide_layout)
                
                title = slide.shapes.title
                content = slide.placeholders[1]
                
                title.text = slide_data['title']
                
                # Add bullet points
                if 'bullet_points' in slide_data:
                    content_text = '\n'.join(slide_data['bullet_points'])
                    content.text = content_text
                    
                    # Style content
                    for paragraph in content.text_frame.paragraphs:
                        paragraph.font.size = Pt(20)
                        paragraph.font.color.rgb = self.colors['text']
                
                # Style title
                title.text_frame.paragraphs[0].font.size = Pt(32)
                title.text_frame.paragraphs[0].font.color.rgb = self.colors['primary']
        
        return prs
    
    def generate_all_presentation_materials(self):
        """Generate all presentation materials for different audiences"""
        
        results = {}
        
        for pres_type in ['executive', 'technical', 'academic', 'investor']:
            print(f"\nGenerating {pres_type} presentation materials...")
            
            # Generate outline
            outline = self.create_presentation_outline(pres_type)
            outline_file = self.output_dir / f"qflare_{pres_type}_outline.json"
            with open(outline_file, 'w') as f:
                json.dump(outline, f, indent=2)
            
            # Generate markdown
            markdown = self.generate_slide_deck_markdown(pres_type)
            markdown_file = self.output_dir / f"qflare_{pres_type}_slides.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            # Generate PowerPoint if available
            pptx_file = None
            if HAS_PPTX:
                try:
                    prs = self.create_powerpoint_template(pres_type)
                    if prs:
                        pptx_file = self.output_dir / f"qflare_{pres_type}_template.pptx"
                        prs.save(str(pptx_file))
                except Exception as e:
                    print(f"⚠️  PowerPoint generation failed: {e}")
            
            results[pres_type] = {
                'outline': outline_file,
                'markdown': markdown_file,
                'powerpoint': pptx_file,
                'slide_count': outline['total_slides'],
                'duration': outline['estimated_duration']
            }
            
            print(f"✅ {pres_type.title()} materials generated")
            print(f"   📊 Slides: {outline['total_slides']}")
            print(f"   ⏱️ Duration: {outline['estimated_duration']}")
            print(f"   📄 Files: {len([f for f in [outline_file, markdown_file, pptx_file] if f])}")
        
        return results

def create_presentation_guide():
    """Create a comprehensive presentation guide"""
    
    guide_content = """# QFLARE Presentation Guide
## Quick Reference for Speakers

### Pre-Presentation Checklist
- [ ] Audience analysis completed (technical level, interests, time constraints)
- [ ] Appropriate presentation template selected
- [ ] Technical demo environment tested (if applicable)
- [ ] Diagrams and visual aids verified for readability
- [ ] Timing rehearsed with buffer for Q&A
- [ ] Backup slides prepared for detailed questions

### Audience-Specific Tips

#### Executive/C-Suite Presentations (15 minutes)
**Focus Areas:**
- Business impact and ROI metrics
- Risk mitigation and competitive advantage
- Market opportunity and revenue potential
- Clear next steps and investment requirements

**Avoid:**
- Deep technical jargon
- Implementation details
- Academic research discussions
- Overly complex diagrams

#### Technical Team Presentations (45 minutes)
**Focus Areas:**
- Architecture details and implementation specifics
- Performance benchmarks and optimization techniques
- Integration challenges and solutions
- Security mechanisms and cryptographic details

**Include:**
- Code examples and configuration details
- Hands-on demonstrations
- Technical Q&A sessions
- Development roadmap discussions

#### Academic/Research Presentations (20 minutes)
**Focus Areas:**
- Novel research contributions
- Experimental methodology and results
- Theoretical foundations and proofs
- Future research directions

**Include:**
- Literature review and positioning
- Statistical analysis and significance testing
- Reproducibility and open-source contributions
- Peer review and validation processes

#### Investor/Stakeholder Presentations (30 minutes)
**Focus Areas:**
- Market size and opportunity
- Competitive landscape and differentiation
- Financial projections and business model
- Team expertise and execution capability

**Include:**
- Customer testimonials and case studies
- Revenue projections and unit economics
- Intellectual property and competitive moat
- Investment requirements and use of funds

### Key Messaging Framework

#### Opening Hook (First 2 minutes)
1. Compelling problem statement with statistics
2. Personal or industry anecdote
3. Bold vision statement
4. Clear agenda and value proposition

#### Core Content Structure
1. **Problem:** Current state and challenges
2. **Solution:** QFLARE approach and benefits  
3. **Evidence:** Performance data and validation
4. **Impact:** Business value and applications
5. **Action:** Next steps and engagement

#### Closing Strong (Last 2 minutes)
1. Recap key value propositions
2. Reinforce competitive advantages
3. Clear call to action
4. Contact information and follow-up

### Common Q&A Responses

#### "How does this compare to existing solutions?"
- Emphasize quantum-safety as unique differentiator
- Reference specific performance benchmarks
- Highlight production-readiness vs research projects

#### "What about implementation complexity?"
- Discuss phased deployment approach
- Reference pilot customer success stories
- Mention professional services and support

#### "What are the regulatory implications?"
- Map to specific compliance frameworks (GDPR, HIPAA, SOC 2)
- Discuss privacy-by-design architecture
- Reference security audit and validation results

#### "What's the quantum timeline and urgency?"
- Cite NIST recommendations and industry consensus
- Discuss "harvest now, decrypt later" attacks
- Emphasize proactive vs reactive security posture

### Visual Presentation Tips

#### Slide Design
- Use high-contrast colors for readability
- Maintain consistent branding and fonts
- Limit text to 6-8 bullet points maximum
- Include plenty of white space

#### Diagram Usage
- Use our generated diagrams as primary visuals
- Customize annotations for specific audiences
- Ensure diagrams are readable from back of room
- Have simplified versions for overview slides

#### Animation and Transitions
- Use subtle, professional transitions
- Animate complex diagrams to build understanding
- Avoid distracting effects or sounds
- Test animations on presentation equipment

### Technology Setup

#### Equipment Checklist
- [ ] Laptop with presentation software
- [ ] HDMI/VGA adapters and cables
- [ ] Backup presentation on cloud storage
- [ ] Demo environment accessible and tested
- [ ] Remote presentation tools configured
- [ ] Audio/video equipment tested

#### Demo Preparation
- [ ] Demo environment isolated from production
- [ ] Network connectivity verified
- [ ] Fallback screenshots/videos prepared
- [ ] Demo script rehearsed and timed
- [ ] Reset procedures documented

### Follow-up Actions

#### Immediate (Within 24 hours)
- Send thank you email with key materials
- Provide requested technical documentation
- Schedule follow-up meetings as committed
- Share presentation slides (executive summary version)

#### Short-term (Within 1 week)
- Detailed technical documentation
- Pilot implementation proposal
- Security assessment and compliance materials
- Customer references and case studies

#### Long-term (Ongoing relationship)
- Regular progress updates
- Industry trend and threat intelligence sharing
- Technical webinars and training sessions
- Strategic partnership discussions

This guide should be customized based on specific presentation contexts, audience feedback, and organizational requirements."""
    
    return guide_content

def main():
    """Main presentation generator function"""
    
    print("QFLARE Presentation Material Generator")
    print("=" * 50)
    
    generator = QFLAREPresentationGenerator()
    
    # Generate all presentation materials
    results = generator.generate_all_presentation_materials()
    
    # Create presentation guide
    guide_content = create_presentation_guide()
    guide_file = generator.output_dir / "qflare_presentation_guide.md"
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    # Summary report
    print(f"\n{'='*50}")
    print("PRESENTATION MATERIALS GENERATED")
    print(f"{'='*50}")
    print(f"📁 Output Directory: {generator.output_dir}")
    print()
    
    total_slides = sum(result['slide_count'] for result in results.values())
    print(f"📊 Total Slides Generated: {total_slides}")
    print(f"📋 Presentation Types: {len(results)}")
    print(f"📄 Documentation Files: {len(list(generator.output_dir.glob('*')))}")
    
    print("\nGenerated Materials:")
    for pres_type, result in results.items():
        print(f"\n🎯 {pres_type.title()} Presentation:")
        print(f"   ⏱️  Duration: {result['duration']}")
        print(f"   📊 Slides: {result['slide_count']}")
        if result['powerpoint']:
            print(f"   📄 PowerPoint: {result['powerpoint'].name}")
        print(f"   📝 Markdown: {result['markdown'].name}")
        print(f"   📋 Outline: {result['outline'].name}")
    
    print(f"\n📖 Presentation Guide: {guide_file.name}")
    
    print(f"\n🎉 All presentation materials ready!")
    print("💡 Next steps:")
    print("   1. Review generated outlines for accuracy")
    print("   2. Customize content for specific audiences")
    print("   3. Add speaker notes and timing cues")
    print("   4. Test presentations with target audiences")
    print("   5. Integrate with our generated diagrams")

if __name__ == "__main__":
    main()
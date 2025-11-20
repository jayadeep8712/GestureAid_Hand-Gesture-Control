# \# GestureAid - Accessibility Hand Gesture Control

# 

# !\[GestureAid Logo](https://img.shields.io/badge/GestureAid-Accessibility\_Tool-blue)

# !\[Python](https://img.shields.io/badge/Python-3.8%2B-green)

# !\[OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-orange)

# !\[MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.18-red)

# 

# \## 🌟 Overview

# 

# GestureAid is an \*\*innovative accessibility tool\*\* that enables computer control through hand gestures. Designed for users with mobility challenges, arthritis, or motor disabilities, it transforms your webcam into a powerful gesture-based input device.

# 

# > \*\*Empowering independence through technology\*\* - Control your computer naturally without touching mouse or keyboard.

# 

# \## 🎯 Key Features

# 

# \### ✨ Accessibility First

# \- \*\*Adaptive Sensitivity\*\* - Automatically adjusts to your movement ability

# \- \*\*Multiple Mobility Profiles\*\* - Pre-configured settings for different ability levels

# \- \*\*Voice \& Visual Feedback\*\* - Clear confirmation of all actions

# \- \*\*No Hardware Required\*\* - Works with any standard webcam

# 

# \### 🎮 Intuitive Gestures

# | Gesture | Action | Usage |

# |---------|--------|-------|

# | 👆 \*\*Index Finger\*\* | Move Cursor | Natural pointer movement |

# | 👌 \*\*Pinch \& Hold\*\* | Left Click | Select, open, activate |

# | ✌️ \*\*Peace Sign\*\* | Right Click | Context menus, options |

# | 🤜 \*\*Fist\*\* | Drag \& Drop | Move files, select text |

# | 👍 \*\*Thumbs Up\*\* | Special Action | Quick app launching |

# 

# \### 🔧 Smart Technology

# \- \*\*Real-time Hand Tracking\*\* - Powered by Google MediaPipe AI

# \- \*\*Adaptive Dwell Times\*\* - Customizable activation delays

# \- \*\*Movement Smoothing\*\* - Reduces hand tremors for precise control

# \- \*\*Gesture Learning\*\* - Improves recognition over time

# 

# \## 🚀 Quick Start

# 

# \### Installation

# 

# 1\. \*\*Clone or Download\*\* the project files

# 2\. \*\*Set up Python environment\*\*:

# &nbsp;  ```bash

# &nbsp;  python -m venv venv

# &nbsp;  venv\\Scripts\\activate  # Windows

# &nbsp;  # source venv/bin/activate  # Mac/Linux

# &nbsp;  ```

# 

# 3\. \*\*Install dependencies\*\*:

# &nbsp;  ```bash

# &nbsp;  pip install -r requirements.txt

# &nbsp;  ```

# 

# \### First Time Setup

# 

# 1\. \*\*Run the application\*\*:

# &nbsp;  ```bash

# &nbsp;  python gestureaid.py

# &nbsp;  ```

# 

# 2\. \*\*Follow the setup wizard\*\*:

# &nbsp;  - Enter your name

# &nbsp;  - Select mobility level (High/Medium/Low)

# &nbsp;  - System auto-configures optimal settings

# 

# 3\. \*\*Position yourself\*\*:

# &nbsp;  - Sit comfortably facing your webcam

# &nbsp;  - Ensure good lighting

# &nbsp;  - Keep hand visible in camera view

# 

# \## 🎮 How to Use

# 

# \### Basic Computer Control

# 

# \#### 🖱️ Cursor Movement

# ```

# 1\. Show your hand to the camera

# 2\. Move index finger to control cursor

# 3\. System smooths movements for precision

# ```

# 

# \#### 📝 Clicking \& Selection

# ```

# 1\. Move cursor to target

# 2\. 👌 Pinch thumb and index finger together

# 3\. Hold for configured dwell time (0.5-1.5s)

# 4\. Release to complete click

# ```

# 

# \#### 📁 File Management

# ```

# 1\. 👆 Move cursor to file

# 2\. 🤜 Make fist to start dragging

# 3\. 👆 Move hand to destination

# 4\. Open hand to drop file

# ```

# 

# \#### 🎯 Right-Click Actions

# ```

# 1\. ✌️ Show peace sign (index + middle fingers up)

# 2\. Hold for dwell time

# 3\. Context menu appears

# ```

# 

# \### Real-World Usage Examples

# 

# \#### 💼 Work Scenario

# ```

# \# Morning Email Routine

# 1\. 👍 Thumbs up → Open email client

# 2\. 👆 Move → Unread messages

# 3\. 👌 Pinch \& hold → Open email

# 4\. 👆 Move → Reply button

# 5\. 👌 Pinch \& hold → Start reply

# 

# \# Document Editing

# 1\. 👆 Move → Word document

# 2\. 👌 Pinch \& hold → Open file

# 3\. 🤜 Fist → Select text block

# 4\. 👆 Move → Highlight desired text

# 5\. Open hand → Complete selection

# ```

# 

# \#### 🏠 Personal Use

# ```

# \# Web Browsing

# 1\. 👍 Thumbs up → Open browser

# 2\. 👆 Move → Search bar

# 3\. 👌 Pinch \& hold → Activate search

# 4\. Type with keyboard

# 5\. 👆 Move → Scroll bar

# 6\. 🤜 Fist → Drag to scroll

# ```

# 

# \## ⚙️ Configuration

# 

# \### Mobility Profiles

# 

# \#### 🟢 High Mobility

# \- Dwell Time: 0.5 seconds

# \- Sensitivity: High

# \- Smoothing: Minimal

# \- Best for: Good hand control, fast movements

# 

# \#### 🟡 Medium Mobility

# \- Dwell Time: 1.0 seconds

# \- Sensitivity: Medium

# \- Smoothing: Moderate

# \- Best for: Some limitations, arthritis

# 

# \#### 🔴 Low Mobility

# \- Dwell Time: 1.5 seconds

# \- Sensitivity: Low

# \- Smoothing: Maximum

# \- Best for: Significant limitations, tremors

# 

# \### Customization

# 

# Edit `gestureaid.py` to modify:

# \- \*\*Dwell times\*\* - Adjust activation delays

# \- \*\*Gesture sensitivity\*\* - Fine-tune detection

# \- \*\*Screen mapping\*\* - Change movement area

# \- \*\*Custom actions\*\* - Add new gesture commands

# 

# \## 🏥 Therapeutic Applications

# 

# \### Rehabilitation Progress

# ```

# Week 1-2: Basic cursor control, single clicks

# Week 3-4: Drag \& drop, right-click functions  

# Week 5-6: Complex tasks, reduced dwell times

# Week 7+: Natural computer use, all gestures

# ```

# 

# \### Usage Tracking

# The system logs:

# \- Session duration

# \- Gesture success rates

# \- Movement patterns

# \- Progress over time

# 

# \## 🔧 Technical Details

# 

# \### System Requirements

# \- \*\*Python\*\*: 3.8 or higher

# \- \*\*Camera\*\*: Standard webcam (720p+ recommended)

# \- \*\*OS\*\*: Windows, macOS, or Linux

# \- \*\*RAM\*\*: 4GB minimum, 8GB recommended

# 

# \### Dependencies

# \- OpenCV - Computer vision processing

# \- MediaPipe - AI hand tracking

# \- NumPy - Mathematical operations

# \- PyAutoGUI - System input control

# 

# \### Architecture

# ```

# GestureAid System

# ├── Hand Tracking (MediaPipe AI)

# ├── Gesture Recognition

# ├── Adaptive Movement System

# ├── User Profile Management

# └── Action Execution

# ```

# 

# \## 🤝 Contributing

# 

# We welcome contributions to improve GestureAid! Areas for development:

# 

# \- \*\*New Gestures\*\* - Additional control options

# \- \*\*Voice Control\*\* - Combined voice+gesture input

# \- \*\*Therapist Dashboard\*\* - Progress monitoring

# \- \*\*Multi-language\*\* - International support

# 

# \## 📞 Support \& Community

# 

# \### Getting Help

# \- \*\*Documentation\*\*: Check this README first

# \- \*\*Issues\*\*: Report bugs on GitHub

# \- \*\*Community\*\*: Join discussions for tips and support

# 

# \### Training Resources

# \- Video tutorials available on our website

# \- Printable gesture reference cards

# \- Therapist training materials

# 

# \## 🙏 Acknowledgments

# 

# \- \*\*Google MediaPipe\*\* for advanced hand tracking AI

# \- \*\*OpenCV community\*\* for computer vision tools

# \- \*\*Testers and therapists\*\* who provided valuable feedback

# \- \*\*Users worldwide\*\* who inspire continued development

# 

# ---

# 

# <div align="center">

# 

# \*\*GestureAid - Empowering Independence Through Technology\*\*

# 

# \*"The power to control your digital world, naturally"\*

# 

# </div>

# 

# \## 🚀 Ready to Start?

# 

# ```bash

# \# Begin your gesture control journey

# python gestureaid.py

# ```

# 

\*\*Transform your computer interaction today!\*\* 🎉


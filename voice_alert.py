import time
import queue
import threading
import sys

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import pythoncom
    HAS_PYTHONCOM = True
except Exception:
    HAS_PYTHONCOM = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except Exception as e:
    HAS_PYTTSX3 = False
    print(f"[ERROR] pyttsx3 module import failed: {e}", file=sys.stderr)

class VoiceAlertManager:
    """
    Non-Blocking Local Small Exam Alarm Ring + Text-to-Speech Voice Alert Engine.
    
    Sequence:
    - Small ring sound (two short high-pitched tones) via winsound.Beep.
    - Short 0.2s pause.
    - Clear spoken warning at ~155 WPM ("Warning. Prohibited object detected.").
    - 4.0s cooldown prevents repeated audio on consecutive frames.
    - 100% non-blocking background queue worker thread (never freezes live camera).
    """

    def __init__(self, cooldown_seconds: float = 4.0, speech_rate: int = 155, enabled: bool = True):
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.speech_rate = speech_rate
        self.last_alert_time = -999.0
        self.alert_queue = queue.Queue()
        self.running = True
        self.speech_msg = "Warning. Restricted object detected."

        # Initialize single background worker thread
        self.worker_thread = threading.Thread(target=self._alert_worker, daemon=True)
        self.worker_thread.start()

    def _speak_text(self, text: str):
        """Executes pyttsx3 text-to-speech with COM thread marshaling on Windows."""
        if not HAS_PYTTSX3:
            print(f"[ERROR] pyttsx3 not installed/available, cannot speak: '{text}'", file=sys.stderr)
            return

        if HAS_PYTHONCOM:
            try:
                pythoncom.CoInitialize()
            except Exception:
                pass

        try:
            if sys.platform == "win32":
                engine = pyttsx3.init('sapi5')
            else:
                engine = pyttsx3.init()
            engine.setProperty('rate', self.speech_rate)  # Slower, natural speaking rate (~155 WPM)
            engine.setProperty('volume', 1.0)

            # Select a clear Windows voice (Zira, David, or English default)
            voices = engine.getProperty('voices')
            if voices:
                for v in voices:
                    v_name = str(v.name).lower()
                    if "zira" in v_name or "david" in v_name or "english" in v_name:
                        engine.setProperty('voice', v.id)
                        break

            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as speech_err:
            print(f"[ERROR] Speech synthesis playback failed: {speech_err}", file=sys.stderr)
        finally:
            if HAS_PYTHONCOM:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def _alert_worker(self):
        """Background worker thread executing small ring sound + clear spoken warning."""
        while self.running:
            try:
                text = self.alert_queue.get(timeout=0.2)
                if text:
                    # Step 1: Small examination-monitoring ring sound (two short tones)
                    if HAS_WINSOUND:
                        try:
                            winsound.Beep(1500, 100)
                            winsound.Beep(1800, 150)
                        except Exception as ring_err:
                            print(f"[ERROR] Alarm ring sound playback failed: {ring_err}", file=sys.stderr)

                    # Short pause between ring sound and speech
                    time.sleep(0.20)

                    # Step 2: Speak warning message clearly
                    self._speak_text(text)

                self.alert_queue.task_done()
            except queue.Empty:
                continue

    def trigger_prohibited_alert(self, detections: list) -> bool:
        """
        Evaluates current detections for PROHIBITED objects (phone, earbuds, smartwatch).
        If found and cooldown period (4.0s) has elapsed, queues non-blocking ring + voice alert.
        """
        if not detections:
            return False

        # Filter for PROHIBITED objects ONLY
        prohibited_items = [
            d for d in detections
            if d.get("status") == "PROHIBITED" or
               d.get("category") == "PROHIBITED" or
               str(d.get("class_name")).lower() in ["phone", "earbuds", "smartwatch"]
        ]

        if not prohibited_items:
            return False

        now = time.time()
        # Enforce 4.0 second cooldown period
        if (now - self.last_alert_time) >= self.cooldown_seconds:
            self.last_alert_time = now
            
            # Clear stale queued messages to prevent backlog
            while not self.alert_queue.empty():
                try:
                    self.alert_queue.get_nowait()
                except queue.Empty:
                    break

            # Queue new alert non-blockingly
            self.alert_queue.put(self.speech_msg)
            print(f"\n[ALERT] [RING + VOICE] Triggered: '{self.speech_msg}'")
            return True

        return False

    def shutdown(self):
        """Stops the worker thread cleanly."""
        self.running = False

if __name__ == "__main__":
    print("==================================================")
    print("TESTING SMALL ALARM RING + TTS VOICE ALERT ENGINE")
    print("==================================================")
    vam = VoiceAlertManager(cooldown_seconds=3.0, speech_rate=155)
    
    # Test allowed object (must NOT trigger)
    allowed_test = [{"class_name": "notebook", "status": "ALLOWED"}]
    t1 = vam.trigger_prohibited_alert(allowed_test)
    print("- Allowed object test (Should be False):", t1)
    
    # Test prohibited object (MUST trigger ring + speech)
    prohibited_test = [{"class_name": "phone", "status": "PROHIBITED"}]
    t2 = vam.trigger_prohibited_alert(prohibited_test)
    print("- Prohibited object test (Should be True):", t2)
    
    time.sleep(3.5) # Wait for ring + speech to finish
    vam.shutdown()
    print("==================================================")
    print("[SUCCESS] Alert Test Completed Successfully!")
    print("==================================================")

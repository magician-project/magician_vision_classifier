#!/usr/bin/python3
"""
Author : "Ammar Qammaz"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Flashes doc/defect.jpg on the desktop when a defect is seen. Same mechanism as the
disconnect pictures of magician_grabber (src/ros_magician_grabber.cpp).
"""

import os
import subprocess
import time

from mvc.paths import repo_root

DEFECT_IMAGE = os.path.join(repo_root(), "doc", "defect.jpg")
POPUP_COOLDOWN_S = 30.0     # measured from when the previous popup appeared
POPUP_MAX_LIFETIME_S = 5.0  # the popup closes itself after 3 s, this is a backstop

# An override-redirect (POPUP) window is not managed by the desktop, so it cannot take
# keyboard focus or disturb other windows. X11 is forced because Wayland does not let a
# client centre its own window; XWayland handles that on Ubuntu 22.04.
_FLASH_SCRIPT = (
    "import sys,gi\n"
    "gi.require_version('Gtk','3.0')\n"
    "from gi.repository import Gtk,Gdk,GLib,GdkPixbuf\n"
    "p=GdkPixbuf.Pixbuf.new_from_file(sys.argv[1])\n"
    "d=Gdk.Display.get_default()\n"
    "g=(d.get_primary_monitor() or d.get_monitor(0)).get_geometry()\n"
    "w=Gtk.Window(type=Gtk.WindowType.POPUP)\n"
    "w.set_accept_focus(False)\n"
    "w.set_focus_on_map(False)\n"
    "w.add(Gtk.Image.new_from_pixbuf(p))\n"
    "w.move(g.x+(g.width-p.get_width())//2,g.y+(g.height-p.get_height())//2)\n"
    "w.show_all()\n"
    "GLib.timeout_add(3000,Gtk.main_quit)\n"
    "Gtk.main()\n"
)


def popup_unavailable_reason(image=DEFECT_IMAGE):
    """Why a popup cannot be shown at all, or None if it can."""
    if not os.environ.get("DISPLAY"):
        return "no DISPLAY in the environment of the node"
    if not os.access(image, os.R_OK):
        return f"cannot read {image}"
    return None


class DefectPopup:
    """At most one popup at a time, and at most one per cooldown.

    Not thread-safe on purpose: only the inference loop may touch this object.
    """

    def __init__(self, logger, image=DEFECT_IMAGE, cooldown_s=POPUP_COOLDOWN_S):
        self._logger = logger
        self._image = image
        self._cooldown_s = cooldown_s
        self._process = None
        self._started = None
        self._warned = False

    def _warn_once(self, message):
        if not self._warned:
            self._logger.warning(message)
            self._warned = True

    def poll(self):
        """Reap a popup that has closed, and enforce the lifetime backstop."""
        if self._process is None:
            return
        status = self._process.poll()
        if status is None:
            if time.monotonic() - self._started > POPUP_MAX_LIFETIME_S:
                self._process.kill()
                self._process.wait()
                self._process = None
            return
        if status > 0:
            self._warn_once("Could not show the defect popup (needs python3-gi and gir1.2-gtk-3.0)")
        self._process = None

    def show(self):
        """Show the popup, unless one is already up or the cooldown is still running."""
        self.poll()
        now = time.monotonic()
        if self._process is not None:
            return
        if self._started is not None and now - self._started < self._cooldown_s:
            return
        reason = popup_unavailable_reason(self._image)
        if reason is not None:
            self._warn_once(f"Cannot show the defect popup: {reason}")
            return

        self._started = now
        try:
            # The system Python, with the venv's library paths and packages kept out. Popen
            # closes inherited fds, so the popup holds nothing of ours open.
            self._process = subprocess.Popen(
                ["/usr/bin/env", "-u", "LD_LIBRARY_PATH", "GDK_BACKEND=x11",
                 "/usr/bin/python3", "-I", "-c", _FLASH_SCRIPT, self._image],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            self._warn_once(f"Could not start the defect popup: {e}")

    def close(self):
        """Close a popup that is still up."""
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None

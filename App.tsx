
import React, { useState, useEffect } from 'react';
import { HashRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  Camera,
  Layers,
  Info,
  Menu,
  X,
  ArrowRight,
  Cpu,
  MessageSquare,
  Volume2,
  CheckCircle2,
  Globe,
  Users
} from 'lucide-react';
import LandingPage from './pages/LandingPage';
import TranslationPage from './pages/TranslationPage';
import WorkflowPage from './pages/WorkflowPage';
import AboutPage from './pages/AboutPage';

const Header = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  const navLinks = [
    { name: 'Home', path: '/' },
    { name: 'Live Translate', path: '/translate' },
    { name: 'Workflow', path: '/workflow' },
    { name: 'About', path: '/about' },
  ];

  return (
    <header className="sticky top-0 z-50 w-full bg-white/80 backdrop-blur-md border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center space-x-2">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Camera className="text-white w-5 h-5" />
            </div>
            <span className="text-xl font-bold text-slate-900 tracking-tight">ISL<span className="text-blue-600">Malayalam</span></span>
          </Link>

          {/* Desktop Nav */}
          <nav className="hidden md:flex space-x-8">
            {navLinks.map((link) => (
              <Link
                key={link.name}
                to={link.path}
                className={`text-sm font-medium transition-colors ${location.pathname === link.path ? 'text-blue-600' : 'text-slate-600 hover:text-blue-600'
                  }`}
              >
                {link.name}
              </Link>
            ))}
          </nav>

          <div className="hidden md:block">
            <Link
              to="/translate"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 shadow-sm transition-all"
            >
              Get Started
            </Link>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-slate-600 hover:text-blue-600 focus:outline-none"
            >
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Nav */}
      {isOpen && (
        <div className="md:hidden bg-white border-b border-slate-200 py-4 px-4 space-y-2">
          {navLinks.map((link) => (
            <Link
              key={link.name}
              to={link.path}
              onClick={() => setIsOpen(false)}
              className="block px-3 py-2 rounded-md text-base font-medium text-slate-700 hover:bg-slate-100 hover:text-blue-600 transition-colors"
            >
              {link.name}
            </Link>
          ))}
          <Link
            to="/translate"
            onClick={() => setIsOpen(false)}
            className="block w-full text-center px-3 py-3 rounded-md text-base font-medium text-white bg-blue-600 hover:bg-blue-700"
          >
            Start Translating
          </Link>
        </div>
      )}
    </header>
  );
};

const Footer = () => (
  <footer className="bg-slate-900 text-slate-300 py-12">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
        <div>
          <h3 className="text-white text-lg font-bold mb-4">ISL Malayalam AI</h3>
          <p className="text-sm leading-relaxed max-w-xs">
            A state-of-the-art gesture recognition system bridging the communication gap for the deaf and mute community using CNN-LSTM and Malayalam TTS.
          </p>
        </div>
        <div>
          <h3 className="text-white text-lg font-bold mb-4">Department Info</h3>
          <ul className="text-sm space-y-2">
            <li>Computer Science & Engineering</li>
            <li>College Project 2026</li>
            <li>Guide: Ms. Madhu Priya</li>
          </ul>
        </div>
        <div>
          <h3 className="text-white text-lg font-bold mb-4">Team Members</h3>
          <ul className="text-sm space-y-2">
            <li>Anal Reji</li>
            <li>Bharath S</li>
            <li>Mahmood Shefin</li>
            <li>Sebastian Saju P</li>
            <li>Vignesh P V</li>
          </ul>
        </div>
      </div>
      <div className="border-t border-slate-800 mt-12 pt-8 text-center text-xs text-slate-500">
        &copy; {new Date().getFullYear()} ISL-to-Malayalam Project Team. All Rights Reserved.
      </div>
    </div>
  </footer>
);

export default function App() {
  return (
    <HashRouter>
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/translate" element={<TranslationPage />} />
            <Route path="/workflow" element={<WorkflowPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </HashRouter>
  );
}

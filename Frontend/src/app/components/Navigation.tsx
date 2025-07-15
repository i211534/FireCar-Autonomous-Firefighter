// src/app/components/Navigation.tsx
import React, { useState } from 'react'
import Link from 'next/link'
import styles from './Navigation.module.css'

const Navigation = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const toggleMenu = () => setIsMenuOpen(!isMenuOpen)

  return (
    <nav className={styles.navbar}>
      <div className={styles.logo}>
        <Link href="/" legacyBehavior><a>FireCar Monitor</a></Link>
      </div>
      <button className={styles.menuToggle} onClick={toggleMenu}>
        ?
      </button>
      <ul className={`${styles.navLinks} ${isMenuOpen ? styles.open : ''}`}>
        <li><Link href="/map" legacyBehavior><a>Room Map</a></Link></li>
        <li><Link href="/camera" legacyBehavior><a>Camera Feed</a></Link></li>
        <li><Link href="/firelogs" legacyBehavior><a>Fire Logs</a></Link></li>
      </ul>
    </nav>
  )
}

export default Navigation

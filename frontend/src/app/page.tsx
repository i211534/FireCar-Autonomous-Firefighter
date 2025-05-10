// src/app/page.tsx
import React from 'react';
import styles from './page.module.css';
import Link from 'next/link';

const HomePage = () => {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Monitor Alerts and View Real-Time Camera Feeds</h1>
      </header>
      <section className={styles.instructions}>
        <p className={styles.subtitle}>Choose an option below to begin:</p>
        <div className={styles.buttons}>
          <Link href="/map" className={`${styles.button} ${styles.roomMapButton}`}>
            View Room Map
          </Link>
          <Link href="/camera" className={`${styles.button} ${styles.cameraFeedButton}`}>
            View Camera Feed
          </Link>
        </div>
      </section>
    </div>
  );
};

export default HomePage;

'use client'

import React, { useEffect, useState } from 'react'
import Navigation from '../components/Navigation'
import styles from './FireLogs.module.css'

type FireLog = {
  id: number
  timestamp: string
  location: string
  severity: string
  suppression_action: string
  resolved: boolean
}

export default function FireLogsPage() {
  const [logs, setLogs] = useState<FireLog[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('http://localhost:5001/firelogs')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data: FireLog[]) => {
        setLogs(data)
        setLoading(false)
      })
      .catch((e) => {
        console.error(e)
        setError('Failed to load logs')
        setLoading(false)
      })
  }, [])

  return (
    <>
      <Navigation />
      <main className={styles.container}>
        <h1>Fire Logs</h1>

        {loading && <p className={styles.message}>Loading�</p>}
        {error && <p className={styles.error}>{error}</p>}
        {!loading && !error && logs.length === 0 && (
          <p className={styles.message}>No fire logs recorded yet.</p>
        )}

        {!loading && !error && logs.length > 0 && (
          <div className={styles.tableWrapper}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Time</th>
                  <th>Location</th>
                  <th>Severity</th>
                  <th>Action</th>
                  <th>Resolved</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => (
                  <tr key={log.id}>
                    <td>{log.id}</td>
                    <td>{new Date(log.timestamp).toLocaleString()}</td>
                    <td>{log.location}</td>
                    <td>{log.severity}</td>
                    <td>{log.suppression_action}</td>
                    <td>{String(log.resolved)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </>
  )
}
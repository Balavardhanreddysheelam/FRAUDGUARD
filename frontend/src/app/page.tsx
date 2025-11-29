"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertTriangle, CheckCircle, RefreshCcw, Zap, XCircle } from "lucide-react"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

interface Transaction {
    id: string
    amount: number
    merchant: string
    timestamp: string
    status: "pending" | "approved" | "declined"
    riskScore: number
    explanation?: string
}

interface ErrorState {
    message: string
    type: "error" | "warning" | "info"
}

export default function Dashboard() {
    const [transactions, setTransactions] = useState<Transaction[]>([])
    const [loading, setLoading] = useState(false)
    const [simulating, setSimulating] = useState(false)
    const [error, setError] = useState<ErrorState | null>(null)
    const [backendStatus, setBackendStatus] = useState<"online" | "offline" | "checking">("checking")

    const checkBackendHealth = async () => {
        try {
            const res = await fetch(`${API_URL}/health`, {
                method: "GET",
                headers: { "Content-Type": "application/json" },
                signal: AbortSignal.timeout(5000) // 5 second timeout
            })
            setBackendStatus(res.ok ? "online" : "offline")
        } catch (err) {
            setBackendStatus("offline")
        }
    }

    const fetchTransactions = async () => {
        // In a real app, this would fetch history from the backend
        // For now, we'll keep the mock history but append new simulations
    }

    const simulateTransaction = async () => {
        setSimulating(true)
        setError(null)
        try {
            const amount = Math.floor(Math.random() * 2000) + 10
            const merchant = ["Amazon", "Unknown Store", "Local Coffee", "Luxury Watch", "Gas Station"][Math.floor(Math.random() * 5)]

            const payload = {
                transaction_id: `TXN_${Date.now()}`,
                user_id: "USER_1",
                amount: amount,
                merchant: merchant,
                timestamp: new Date().toISOString()
            }

            const controller = new AbortController()
            const timeoutId = setTimeout(() => controller.abort(), 30000) // 30 second timeout

            const res = await fetch(`${API_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal: controller.signal
            })

            clearTimeout(timeoutId)

            if (!res.ok) {
                const errorData = await res.json().catch(() => ({ detail: "Unknown error" }))
                throw new Error(errorData.detail || `API returned ${res.status}`)
            }

            const data = await res.json()

            // Validate response data
            if (!data.transaction_id || typeof data.risk_score !== "number") {
                throw new Error("Invalid response format from API")
            }

            const newTxn: Transaction = {
                id: data.transaction_id,
                amount: amount,
                merchant: merchant,
                timestamp: new Date().toISOString(),
                status: data.is_fraud ? "declined" : "approved",
                riskScore: Math.max(0, Math.min(1, data.risk_score)), // Clamp to [0, 1]
                explanation: data.explanation || "No explanation provided"
            }

            setTransactions(prev => [newTxn, ...prev])
            await checkBackendHealth() // Update backend status
        } catch (error) {
            console.error("Simulation failed:", error)
            const errorMessage = error instanceof Error 
                ? error.message 
                : "Failed to connect to backend. Please check if the API is running."
            
            setError({
                message: errorMessage,
                type: "error"
            })
            
            // Auto-dismiss error after 5 seconds
            setTimeout(() => setError(null), 5000)
        } finally {
            setSimulating(false)
        }
    }

    useEffect(() => {
        // Initial mock data
        setTransactions([
            {
                id: "TXN_12345",
                amount: 1250.00,
                merchant: "Unknown Electronics Store",
                timestamp: new Date().toISOString(),
                status: "pending",
                riskScore: 0.85,
                explanation: "High transaction amount for this user pattern. Merchant category is high risk."
            },
            {
                id: "TXN_12346",
                amount: 25.50,
                merchant: "Local Coffee Shop",
                timestamp: new Date().toISOString(),
                status: "approved",
                riskScore: 0.05,
                explanation: "Transaction fits normal user behavior."
            },
            {
                id: "TXN_12347",
                amount: 4500.00,
                merchant: "Luxury Watches Online",
                timestamp: new Date().toISOString(),
                status: "declined",
                riskScore: 0.95,
                explanation: "Unusual location and very high amount. Device fingerprint mismatch."
            }
        ])
        
        // Check backend health on mount
        checkBackendHealth()
        
        // Periodic health check every 30 seconds
        const healthCheckInterval = setInterval(checkBackendHealth, 30000)
        return () => clearInterval(healthCheckInterval)
    }, [])

    return (
        <div className="min-h-screen bg-gray-50 p-8">
            <div className="mx-auto max-w-5xl space-y-8">
                {/* Error Banner */}
                {error && (
                    <div className={`rounded-lg border p-4 ${
                        error.type === "error" ? "bg-red-50 border-red-200 text-red-800" :
                        error.type === "warning" ? "bg-yellow-50 border-yellow-200 text-yellow-800" :
                        "bg-blue-50 border-blue-200 text-blue-800"
                    }`}>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <XCircle className="h-4 w-4" />
                                <p className="font-medium">{error.message}</p>
                            </div>
                            <button onClick={() => setError(null)} className="text-current opacity-70 hover:opacity-100">
                                <XCircle className="h-4 w-4" />
                            </button>
                        </div>
                    </div>
                )}

                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight text-gray-900">FraudGuard Dashboard</h1>
                        <p className="text-gray-500">Real-time financial fraud monitoring</p>
                    </div>
                    <div className="flex gap-2">
                        <Button onClick={simulateTransaction} disabled={simulating || backendStatus === "offline"} variant="default">
                            <Zap className={`mr-2 h-4 w-4 ${simulating ? "animate-pulse" : ""}`} />
                            {simulating ? "Processing..." : "Simulate Transaction"}
                        </Button>
                        <Button onClick={() => window.location.reload()} variant="outline">
                            <RefreshCcw className="mr-2 h-4 w-4" />
                            Reset
                        </Button>
                    </div>
                </div>

                <div className="grid gap-4 md:grid-cols-3">
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Total Transactions</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{transactions.length}</div>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">Fraud Alerts</CardTitle>
                            <AlertTriangle className="h-4 w-4 text-red-500" />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-red-600">
                                {transactions.filter(t => t.riskScore > 0.5).length}
                            </div>
                        </CardContent>
                    </Card>
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">System Status</CardTitle>
                            {backendStatus === "online" ? (
                                <CheckCircle className="h-4 w-4 text-green-500" />
                            ) : backendStatus === "offline" ? (
                                <XCircle className="h-4 w-4 text-red-500" />
                            ) : (
                                <RefreshCcw className="h-4 w-4 text-yellow-500 animate-spin" />
                            )}
                        </CardHeader>
                        <CardContent>
                            <div className={`text-2xl font-bold ${
                                backendStatus === "online" ? "text-green-600" :
                                backendStatus === "offline" ? "text-red-600" :
                                "text-yellow-600"
                            }`}>
                                {backendStatus === "online" ? "Online" :
                                 backendStatus === "offline" ? "Offline" :
                                 "Checking..."}
                            </div>
                            <p className="text-xs text-muted-foreground">
                                Backend: {backendStatus === "online" ? "Connected" :
                                          backendStatus === "offline" ? "Disconnected" :
                                          "Checking..."}
                            </p>
                        </CardContent>
                    </Card>
                </div>

                <Card>
                    <CardHeader>
                        <CardTitle>Recent Transactions</CardTitle>
                        <CardDescription>Live stream of processed transactions.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        {transactions.length === 0 ? (
                            <div className="text-center py-8 text-muted-foreground">
                                <p>No transactions yet. Click "Simulate Transaction" to get started.</p>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {transactions.map((txn) => (
                                <div key={txn.id} className="flex items-center justify-between rounded-lg border p-4 shadow-sm transition-all hover:bg-gray-50">
                                    <div className="space-y-1">
                                        <p className="font-medium leading-none">{txn.merchant}</p>
                                        <p className="text-sm text-muted-foreground">{txn.id} • {new Date(txn.timestamp).toLocaleTimeString()}</p>
                                        {txn.explanation && (
                                            <p className="text-xs text-blue-600 mt-1">🤖 AI: {txn.explanation}</p>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <div className="text-right">
                                            <p className="font-medium">${txn.amount.toFixed(2)}</p>
                                            <Badge
                                                variant={txn.riskScore > 0.8 ? "destructive" : txn.riskScore > 0.5 ? "secondary" : "outline"}
                                                className={txn.riskScore > 0.8 ? "bg-red-100 text-red-800 hover:bg-red-200" : txn.riskScore > 0.5 ? "bg-yellow-100 text-yellow-800 hover:bg-yellow-200" : "bg-green-100 text-green-800 hover:bg-green-200"}
                                            >
                                                Risk: {(txn.riskScore * 100).toFixed(0)}%
                                            </Badge>
                                        </div>
                                    </div>
                                </div>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}

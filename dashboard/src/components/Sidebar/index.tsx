import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
    Drawer,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    IconButton,
    Divider,
    useTheme,
} from '@mui/material';
import {
    Dashboard as DashboardIcon,
    ShowChart as ShowChartIcon,
    Settings as SettingsIcon,
    Assessment as AssessmentIcon,
    Timeline as TimelineIcon,
    ChevronLeft as ChevronLeftIcon,
} from '@mui/icons-material';

interface SidebarProps {
    open: boolean;
    onClose: () => void;
}

const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Trades', icon: <ShowChartIcon />, path: '/trades' },
    { text: 'Strategies', icon: <TimelineIcon />, path: '/strategies' },
    { text: 'Performance', icon: <AssessmentIcon />, path: '/performance' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
];

const Sidebar: React.FC<SidebarProps> = ({ open, onClose }) => {
    const theme = useTheme();
    const navigate = useNavigate();
    const location = useLocation();

    const handleNavigation = (path: string) => {
        navigate(path);
    };

    return (
        <Drawer
            variant="permanent"
            open={open}
            sx={{
                width: open ? 240 : 65,
                flexShrink: 0,
                whiteSpace: 'nowrap',
                boxSizing: 'border-box',
                '& .MuiDrawer-paper': {
                    width: open ? 240 : 65,
                    transition: theme.transitions.create('width', {
                        easing: theme.transitions.easing.sharp,
                        duration: theme.transitions.duration.enteringScreen,
                    }),
                    overflowX: 'hidden',
                    backgroundColor: theme.palette.background.paper,
                    borderRight: `1px solid ${theme.palette.divider}`,
                },
            }}
        >
            {/* Sidebar Header */}
            <List>
                <ListItem
                    sx={{
                        display: 'flex',
                        justifyContent: open ? 'flex-end' : 'center',
                        px: 2.5,
                    }}
                >
                    <IconButton onClick={onClose}>
                        <ChevronLeftIcon />
                    </IconButton>
                </ListItem>
            </List>
            <Divider />

            {/* Navigation Items */}
            <List>
                {menuItems.map((item) => (
                    <ListItem
                        button
                        key={item.text}
                        onClick={() => handleNavigation(item.path)}
                        selected={location.pathname === item.path}
                        sx={{
                            minHeight: 48,
                            justifyContent: open ? 'initial' : 'center',
                            px: 2.5,
                            '&.Mui-selected': {
                                backgroundColor: theme.palette.action.selected,
                                '&:hover': {
                                    backgroundColor: theme.palette.action.selected,
                                },
                            },
                        }}
                    >
                        <ListItemIcon
                            sx={{
                                minWidth: 0,
                                mr: open ? 3 : 'auto',
                                justifyContent: 'center',
                            }}
                        >
                            {item.icon}
                        </ListItemIcon>
                        <ListItemText
                            primary={item.text}
                            sx={{
                                opacity: open ? 1 : 0,
                                transition: theme.transitions.create('opacity', {
                                    duration: theme.transitions.duration.enteringScreen,
                                }),
                            }}
                        />
                    </ListItem>
                ))}
            </List>
        </Drawer>
    );
};

export default Sidebar; 